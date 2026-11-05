"""Algorithm/data contracts and short synthetic execution checks (no MRI claims)."""
from dataclasses import replace
from pathlib import Path
import ast
import csv
import json
import tempfile
import unittest
import numpy as np
import torch
from experiments.rgap.engine import Config, run, seed_all
from experiments.rgap.models import make_model
from experiments.rgap.physics import (channels, combine, fft2c, ifft2c, make_mask,
                                      residual_gate, update_gate)
from experiments.rgap.external import validate_output


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.backends.nnpack.enabled = False

    def test_uploaded_gate_exactly(self):
        original = Path(__file__).resolve().parent/"fixtures/Residual_Guided_original.py"
        tree=ast.parse(original.read_text())
        names={"fft_with_shifts","ifft_with_shifts","ksp_and_mps_to_gt","residual_to_gate"}
        module=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[])
        env={"torch":torch,"fft":torch.fft}
        exec(compile(module,"uploaded_helpers","exec"),env)
        torch.manual_seed(41)
        residual=torch.randn(3,16,16,dtype=torch.complex64)
        sens=torch.randn(3,16,16,dtype=torch.complex64)
        actual=residual_gate(residual,sens)
        expected=env["residual_to_gate"](residual[None],sens[None])
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)

    def test_gate_boundaries_and_ema(self):
        sens=torch.ones(1,8,8,dtype=torch.complex64)
        zero=torch.zeros_like(sens)
        self.assertEqual(residual_gate(zero,sens).abs().max().item(),0)
        constant=fft2c(torch.ones_like(sens))
        gate=residual_gate(constant,sens)
        torch.testing.assert_close(gate,torch.ones_like(gate),atol=1e-6,rtol=1e-6)
        smoothed=update_gate(torch.ones_like(gate),torch.zeros_like(gate),.05,.05)
        torch.testing.assert_close(smoothed,torch.full_like(gate,.95))
        direct=update_gate(torch.ones_like(gate),torch.zeros_like(gate),1,.05)
        torch.testing.assert_close(direct,torch.full_like(gate,.05))
        image=torch.zeros_like(sens); image[0,2,3]=1e10
        gate=residual_gate(fft2c(image),sens)
        self.assertTrue(torch.isfinite(gate).all()); self.assertLessEqual(float(gate.max()),1)

    def test_randn_covariance_not_uniform(self):
        rng=torch.Generator().manual_seed(5)
        noise=torch.randn(200000,2,generator=rng)
        self.assertLess(abs(float(noise.mean())),.01)
        self.assertLess(abs(float(noise.var())-1),.015)
        g=torch.tensor([.3,.8]); sigma=2.0
        torch.testing.assert_close((sigma*g*noise).var(0),sigma**2*g**2,rtol=.02,atol=.003)

    def test_mask_exact_source_and_count(self):
        original=Path(__file__).resolve().parent/"fixtures/Residual_Guided_original.py"
        tree=ast.parse(original.read_text())
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="make_vdrs_mask")
        env={"np":np}
        exec(compile(ast.Module(body=[node],type_ignores=[]),"source_mask","exec"),env)
        for acc,count in [(4,79),(8,40)]:
            mask,meta=make_mask(320,320,acc,0)
            expected=env["make_vdrs_mask"](320,320,int(320/acc),int(320*.32/acc)).astype(np.float32)
            np.testing.assert_array_equal(mask,expected)
            self.assertEqual(meta["sampled_lines"],count)

    def test_ortho_adjoint(self):
        torch.manual_seed(7)
        x=torch.randn(16,16,dtype=torch.complex64)
        s=torch.randn(3,16,16,dtype=torch.complex64)
        k=torch.randn(3,16,16,dtype=torch.complex64)
        mask=torch.randint(0,2,(16,16)).float()
        left=(fft2c(s*x,"ortho")*mask).conj().mul(k).sum()
        right=x.conj().mul(combine(k*mask,s,"ortho")).sum()
        torch.testing.assert_close(left,right,rtol=1e-5,atol=1e-5)

    def test_case_lists_have_fifteen_npz_files(self):
        path=Path(__file__).resolve().parents[1]/"configs/cases.json"
        data=json.loads(path.read_text())
        self.assertEqual(set(data),{"knee","t1","t2"})
        for names in data.values():
            self.assertEqual(len(names),15)
            self.assertEqual(len(set(names)),15)
            self.assertTrue(all(n.endswith(".npz") for n in names))

    def test_rescat_shared_initialization(self):
        seed_all(21); base=make_model("tiny")
        seed_all(21); cat=make_model("tiny",True)
        x=torch.randn(1,2,16,16)
        torch.testing.assert_close(base(x),cat(torch.cat([x,torch.zeros_like(x)],1)))
        self.assertEqual(cat.inc[0].in_channels,4)

    def test_native_end_to_end_and_selection(self):
        cfg=Config(steps=3,size=32,backbone="tiny",synthetic=True,device="cpu",save_every=1)
        with tempfile.TemporaryDirectory() as root:
            for method in ("rgap","dip","no_ema","no_tv","joint_opt","rescat","uniform_noise","shuffled_gate","global_gate"):
                c=replace(cfg,method=method)
                result=run(c,None,root)
                folder=Path(result["output_path"])
                with open(folder/"trajectory.csv") as f: rows=list(csv.DictReader(f))
                best=max(rows,key=lambda r:float(r["psnr"]))
                self.assertEqual(result["best"]["step"],int(best["step"]))
                self.assertEqual(result["best"]["ssim"],float(best["ssim"]))
                self.assertEqual(result["final"]["step"],3)
                self.assertEqual(len(rows),3)
                self.assertTrue(np.isfinite(result["best"]["psnr"]))
                self.assertTrue(result["not_publication_run"])
                if method=="no_tv": self.assertEqual(result["effective_tv"],0)
                if method=="rgap":
                    reference=torch.as_tensor(np.load(folder/"reconstructions.npz")["reference"])
                    verified=validate_output(folder,replace(c,method="sg_dip"),reference,
                        {"implementation":"TEST_ONLY_contract_check","version":"test"})
                    self.assertEqual(verified["best"],result["best"])
                    repeated=run(c,None,Path(root)/"repeat")
                    self.assertEqual(repeated["best"],result["best"])
                    self.assertEqual(repeated["final"],result["final"])


if __name__=="__main__":
    unittest.main()
