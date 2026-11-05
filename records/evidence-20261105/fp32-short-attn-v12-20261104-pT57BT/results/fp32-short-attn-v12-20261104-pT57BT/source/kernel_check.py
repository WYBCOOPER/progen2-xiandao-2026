from check_rope import main as rope
from check_short import main as short
if __name__=="__main__":
    rope();short()
    print("KERNEL_CHECK=PASS",flush=True)
