from check_attention import main as attention_check
from check_rope import main as rope_check

if __name__ == "__main__":
    attention_check()
    rope_check()
    print("KERNEL_CHECK=PASS", flush=True)
