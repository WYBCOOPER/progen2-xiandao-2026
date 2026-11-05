import torch
from check_compiled_rope import main as rope
from check_short import tensors,reference
from models.progen.kernels import short_attention

def main():
    rope()
    f=torch.compile(short_attention,mode="max-autotune-no-cudagraphs",fullgraph=True,dynamic=True)
    count=0
    with torch.inference_mode():
        for b in (1,2,8):
            for s in (32,64,128):
                args=tensors(b,s,1.,"cuda")
                for _ in range(2):
                    y=f(*args)
                    torch.testing.assert_close(y,reference(*args),rtol=2e-4,atol=2e-4)
                    count+=1
    print(f"COMPILED_SHORT_CHECK=PASS cases={count}",flush=True)
    print("COMPILED_KERNEL_CHECK=PASS",flush=True)

if __name__=="__main__":main()
