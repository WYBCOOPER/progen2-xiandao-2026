import torch
from models.progen.kernels import greedy_token
from solution import _greedy_local_index

def cases(device):
    torch.manual_seed(113)
    allowed = torch.randperm(32, device=device)[:20].contiguous()
    for batch in (1, 2, 4, 8):
        for length in (1, 32):
            # Also cover a non-unit vocabulary stride.
            for stride in (1, 2):
                for case in ("random", "ties", "neg_inf", "pos_inf", "nan", "all_nan", "near_tie"):
                    storage = torch.randn(batch, length, 32 * stride, device=device)
                    logits = storage[:, -1, ::stride]
                    if case == "ties": logits.fill_(3.0)
                    if case == "neg_inf": logits.fill_(-float("inf"))
                    if case == "pos_inf": logits[:, allowed[3]] = float("inf"); logits[:, allowed[9]] = float("inf")
                    if case == "nan": logits[:, allowed[4]] = float("nan"); logits[:, allowed[13]] = float("nan")
                    if case == "all_nan": logits.fill_(float("nan"))
                    if case == "near_tie":
                        logits.fill_(-1.0)
                        logits[:, allowed[0]] = 1.0
                        logits[:, allowed[19]] = torch.nextafter(torch.tensor(1.,device=device),torch.tensor(2.,device=device))
                    expected = allowed.index_select(0, _greedy_local_index(logits.index_select(-1, allowed))).unsqueeze(-1)
                    yield logits, allowed, expected, (batch, length, stride, case)

def main():
    assert torch.cuda.is_available() and torch.version.hip, "DCU/HIP required"
    count = 0
    with torch.inference_mode():
        for logits, allowed, expected, label in cases("cuda"):
            before=logits.clone(); ids_before=allowed.clone()
            actual=greedy_token(logits, allowed)
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            torch.testing.assert_close(logits,before,rtol=0,atol=0,equal_nan=True)
            assert torch.equal(allowed,ids_before)
            count+=1
    print(f"GREEDY_CHECK=PASS cases={count}",flush=True)

if __name__=="__main__":main()
