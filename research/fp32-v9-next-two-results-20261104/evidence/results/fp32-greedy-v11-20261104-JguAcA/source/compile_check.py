import torch
from check_compiled_rope import main as check_rope
from check_greedy import cases
from models.progen.kernels import greedy_token

def main():
    check_rope()
    compiled=torch.compile(greedy_token,mode="max-autotune-no-cudagraphs",fullgraph=True,dynamic=True)
    count=0
    with torch.inference_mode():
        # Two passes test changing data without changing the generated code.
        for _ in range(2):
            for logits,allowed,expected,label in cases("cuda"):
                actual=compiled(logits,allowed)
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                count+=1
    torch.cuda.synchronize()
    print(f"COMPILED_GREEDY_CHECK=PASS cases={count}",flush=True)
    print("COMPILED_KERNEL_CHECK=PASS",flush=True)

if __name__=="__main__":main()
