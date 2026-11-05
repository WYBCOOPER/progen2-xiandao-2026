"""DCU prerequisite check, outside official test timing; not a speed benchmark."""
import json
import torch
from models.progen.kernels import decode_attention


def main():
    assert torch.cuda.is_available(), "DCU is not visible"
    assert torch.version.hip is not None, "This package targets the DCU/HIP environment"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(812)
    records=[]
    with torch.inference_mode():
        for batch in (1,4,8):
            # Noncontiguous query strides and valid slices of a static cache.
            q=torch.randn(batch,1,16,96,device="cuda").transpose(1,2)
            k=torch.randn(batch,16,1024,96,device="cuda")
            v=torch.randn_like(k)
            for length in (1,2,31,32,33,63,64,65,127,128,129,287,511,512,543,576,1023,1024):
                for scale in (1.0,8.0):
                    query=q*scale
                    actual=decode_attention(query,k[:,:,:length],v[:,:,:length])
                    scores=query@k[:,:,:length].transpose(-1,-2)
                    reference=(scores/torch.sqrt(torch.tensor(96.,device="cuda"))).softmax(-1)@v[:,:,:length]
                    assert actual.dtype==torch.float32 and actual.shape==reference.shape
                    assert torch.isfinite(actual).all(),(batch,length,scale,"nonfinite")
                    torch.testing.assert_close(actual,reference,rtol=2e-4,atol=2e-4)
                    records.append({"B":batch,"S":length,"q_scale":scale,
                                    "max_abs_error":(actual-reference).abs().max().item()})
    torch.cuda.synchronize()
    print("KERNEL_CHECK_DETAILS="+json.dumps(records))
    print("KERNEL_CHECK=PASS",flush=True)


if __name__=="__main__":main()
