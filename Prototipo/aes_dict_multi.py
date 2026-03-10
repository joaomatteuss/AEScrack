import argparse
import base64
import hashlib
import sys
import multiprocessing as mp
from typing import Optional, Tuple, List
from Crypto.Cipher import AES
from Crypto.Protocol.KDF import PBKDF2, scrypt as pycrypto_scrypt
from Crypto.Hash import SHA256
from tqdm import tqdm

AES_KEY_LEN = 32
BLOCK = 16
DEFAULT_PREFIX = "Local de descarga:"


def pkcs7_unpad(b: bytes):
    if not b:
        return None
    pad = b[-1]
    if pad < 1 or pad > BLOCK:
        return None
    if b[-pad:] != bytes([pad]) * pad:
        return None
    return b[:-pad]


def key_raw_zero(passphrase: str) -> bytes:
    b = passphrase.encode("utf-8", errors="ignore")
    return (b + b"\x00" * AES_KEY_LEN)[:AES_KEY_LEN]


def key_raw_spaces(passphrase: str) -> bytes:
    b = passphrase.encode("utf-8", errors="ignore")
    return (b + b" " * AES_KEY_LEN)[:AES_KEY_LEN]


def key_repeat(passphrase: str) -> bytes:
    b = passphrase.encode("utf-8", errors="ignore")
    if not b:
        return b"\x00" * AES_KEY_LEN
    out = bytearray()
    i = 0
    while len(out) < AES_KEY_LEN:
        out.append(b[i % len(b)])
        i += 1
    return bytes(out[:AES_KEY_LEN])


def key_md5_repeat(passphrase: str) -> bytes:
    m = hashlib.md5(passphrase.encode("utf-8", errors="ignore")).digest()
    return (m + m)[:AES_KEY_LEN]


def key_sha1_extend(passphrase: str) -> bytes:
    s1 = hashlib.sha1(passphrase.encode("utf-8", errors="ignore")).digest()
    return (s1 + s1[:12])[:AES_KEY_LEN]


def key_sha256(passphrase: str) -> bytes:
    return hashlib.sha256(passphrase.encode("utf-8", errors="ignore")).digest()[
        :AES_KEY_LEN
    ]


def key_sha512(passphrase: str) -> bytes:
    return hashlib.sha512(passphrase.encode("utf-8", errors="ignore")).digest()[
        :AES_KEY_LEN
    ]


def key_md5hex_repeat(passphrase: str) -> bytes:
    md5hex = (
        hashlib.md5(passphrase.encode("utf-8", errors="ignore"))
        .hexdigest()
        .encode("utf-8")
    )
    return (md5hex * 2)[:AES_KEY_LEN]


def key_utf16le(passphrase: str) -> Optional[bytes]:
    try:
        b16 = passphrase.encode("utf-16le")
        return (b16 + b"\x00" * AES_KEY_LEN)[:AES_KEY_LEN]
    except Exception:
        return None


def key_hex(passphrase: str) -> Optional[bytes]:
    try:
        raw = bytes.fromhex(passphrase)
    except Exception:
        return None
    if len(raw) == AES_KEY_LEN:
        return raw
    if len(raw) > AES_KEY_LEN:
        return raw[:AES_KEY_LEN]
    return raw + b"\x00" * (AES_KEY_LEN - len(raw))


def key_scrypt(
    passphrase: str, salt: str = "salt", n: int = 2**14, r: int = 8, p: int = 1
) -> bytes | Tuple[bytes, ...]:
    # pycryptodome scrypt
    try:
        return pycrypto_scrypt(passphrase, key_len=32, salt=salt, N=n, r=r, p=p)
    except TypeError:
        return pycrypto_scrypt(passphrase, salt, n, r, p, AES_KEY_LEN)


def key_pbkdf2(passphrase: str, salt: bytes = b"", iterations: int = 1000) -> bytes:
    return PBKDF2(
        passphrase,
        salt,
        dkLen=AES_KEY_LEN,
        count=iterations,
        hmac_hash_module=SHA256,
    )


# generator that yields simple (desc, key) variants (non-PBKDF2/non-Scrypt)
def key_variants_simple(pw: str):
    yield ("raw-zero", key_raw_zero(pw))
    yield ("raw-spaces", key_raw_spaces(pw))
    yield ("repeat", key_repeat(pw))
    yield ("md5-repeat", key_md5_repeat(pw))
    yield ("sha1-extend", key_sha1_extend(pw))
    yield ("sha256", key_sha256(pw))
    yield ("sha512", key_sha512(pw))
    yield ("md5hex-repeat", key_md5hex_repeat(pw))
    k16 = key_utf16le(pw)
    if k16:
        yield ("utf16le", k16)
    kh = key_hex(pw)
    if kh:
        yield ("hex", kh)


# PBKDF2 variants generator (salts: list of bytes, iters: list of ints)
def pbkdf2_variants(pw: str, salts: List[bytes], iters: List[int]):
    pw.encode("utf-8", errors="ignore")
    for s in salts:
        for it in iters:
            try:
                k = key_pbkdf2(pw, salt=s, iterations=it)
            except Exception:
                continue
            yield (f"pbkdf2(salt={s},iter={it})", k)


# scrypt variants generator (salts: list of bytes, params: list of (n,r,p) tuples)
def scrypt_variants(pw: str, salts: List[bytes], params: List[Tuple[int, int, int]]):
    for s in salts:
        for n, r, p in params:
            try:
                k = key_scrypt(pw, salt=s.decode("utf-8"), n=n, r=r, p=p)
            except Exception:
                continue
            yield (
                f"scrypt(salt={s.decode('utf-8',errors='ignore')},N={n},r={r},p={p})",
                k,
            )


def blocks_needed_for_prefix(prefix_str: str):
    L = len(prefix_str.encode("utf-8"))
    return (L + BLOCK - 1) // BLOCK


def check_key_prefix(
    cipher_bytes: bytes, key: bytes, prefix_bytes: bytes, blocks_to_check: int
):
    if len(key) != AES_KEY_LEN:
        return False, None
    cipher = AES.new(key, AES.MODE_ECB)
    num = blocks_to_check * BLOCK
    if len(cipher_bytes) < num:
        return False, None
    ct_part = cipher_bytes[:num]
    pt_part = cipher.decrypt(ct_part)
    if pt_part.startswith(prefix_bytes):
        if len(cipher_bytes) % BLOCK != 0:

            return True, cipher.decrypt(
                cipher_bytes[: (len(cipher_bytes) // BLOCK) * BLOCK]
            )
        full_pt = cipher.decrypt(cipher_bytes)
        unp = pkcs7_unpad(full_pt)
        return True, (full_pt if unp is None else unp)
    return False, None


def worker(
    task: Tuple[
        str, bytes, bytes, int, List[bytes], List[int], List[Tuple[int, int, int]]
    ],
):

    (
        candidate,
        cipher_bytes,
        prefix_bytes,
        blocks_to_check,
        salts,
        iters,
        scrypt_params,
    ) = task

    # simple variants
    for desc, key in key_variants_simple(candidate):
        ok, pt = check_key_prefix(cipher_bytes, key, prefix_bytes, blocks_to_check)
        if ok:
            return (candidate, desc, key.hex(), pt)

    # pbkdf2 variants
    for desc, key in pbkdf2_variants(candidate, salts, iters):
        ok, pt = check_key_prefix(cipher_bytes, key, prefix_bytes, blocks_to_check)
        if ok:
            return (candidate, desc, key.hex(), pt)

    # scrypt variants
    for desc, key in scrypt_variants(candidate, salts, scrypt_params):
        if type(key) != bytes:
            continue
        ok, pt = check_key_prefix(cipher_bytes, key, prefix_bytes, blocks_to_check)
        if ok and (type(key) == bytes):
            return (candidate, desc, key.hex(), pt)

    return None


def load_candidates(wordlist_arg: str) -> list[str | None]:
    if wordlist_arg == "-":
        return [line.strip() for line in sys.stdin if line.strip()]
    else:
        with open(wordlist_arg, "r", encoding="utf-8", errors="ignore") as f:
            return [l.strip() for l in f if l.strip()]


def parse_args():
    ap = argparse.ArgumentParser(
        description="AES-256-ECB dictionary/PBKDF2/Scrypt tester (known prefix check)."
    )
    ap.add_argument("--cipher", help="ciphertext Base64 string")
    ap.add_argument("--cipher-file", help="file containing Base64 ciphertext")
    ap.add_argument(
        "--wordlist",
        required=True,
        help="path to wordlist (one per line) or '-' for stdin",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=max(1, mp.cpu_count() - 1),
        help="number of worker processes",
    )
    ap.add_argument(
        "--salt",
        action="append",
        help="candidate salt for PBKDF2/scrypt (can repeat). e.g. --salt '' --salt salt",
    )
    ap.add_argument(
        "--iter",
        type=int,
        action="append",
        help="iteration count for PBKDF2 (can repeat). e.g. --iter 100 --iter 1000",
    )
    ap.add_argument(
        "--prefix", default=DEFAULT_PREFIX, help="known plaintext prefix to check"
    )
    ap.add_argument(
        "--stop-on-find", action="store_true", help="stop early when a match is found"
    )
    return ap.parse_args()


def main():
    args = parse_args()

    if args.cipher:
        cipher_b64 = args.cipher.strip()
    elif args.cipher_file:
        with open(args.cipher_file, "r", encoding="utf-8") as f:
            cipher_b64 = f.read().strip()
    else:
        print("Provide --cipher or --cipher-file")
        sys.exit(1)

    try:
        cipher_bytes = base64.b64decode(cipher_b64)
    except Exception as e:
        print("[!] Invalid Base64 ciphertext:", e)
        sys.exit(1)

    prefix_bytes = args.prefix.encode("utf-8")
    blocks_to_check = blocks_needed_for_prefix(args.prefix)

    salts = (
        [s.encode("utf-8") for s in args.salt]
        if args.salt
        else [b"", b"salt", b"app", b"user"]
    )
    iters = args.iter if args.iter else [100, 1000]
    scrypt_params = [(2**14, 8, 1), (2**12, 8, 1)]

    candidates = load_candidates(args.wordlist)
    total = len(candidates)
    if total == 0:
        print("[!] No candidates loaded.")
        sys.exit(1)

    print(
        f"[+] Loaded {total} candidates. Workers: {args.workers}. Prefix blocks: {blocks_to_check}"
    )
    tasks_gen = (
        (c, cipher_bytes, prefix_bytes, blocks_to_check, salts, iters, scrypt_params)
        for c in candidates
    )

    pool = mp.Pool(processes=args.workers)

    try:
        with open("results.txt", "w", encoding="utf-8") as fh:
            fh.write("AES Bruteforce Results \n\n")  # simple text

            for res in tqdm(
                pool.imap_unordered(worker, tasks_gen, chunksize=64), total=total
            ):

                if res:
                    candidate, desc, keyhex, pt = res
                    if type(pt) != bytes:
                        continue
                    try:
                        preview = pt.decode("utf-8", errors="replace")[:500]
                    except UnicodeDecodeError:
                        preview = repr(pt)[:200]

                    fh.write(f"[FOUND]\n")
                    fh.write(f"Password: {candidate}\n")
                    fh.write(f"Variant: {desc}\n")
                    fh.write(f"Key (hex): {keyhex}\n")
                    fh.write(f"Plaintext preview:\n{preview}\n")
                    fh.flush()

                    print(
                        f"\n[FOUND] candidate: {candidate} variant: {desc} key: {keyhex}"
                    )
                    print("plaintext preview:", preview)

                    if args.stop_on_find:
                        pool.terminate()
                        pool.join()
                        print(
                            "[*] Stopped on first find. Results written to results.txt"
                        )
                        return

            print("[*] Completed search. Results stored in results.txt")

    except KeyboardInterrupt:
        print("\n[!] Interrupted by user. Terminating workers...")
        pool.terminate()
        pool.join()
    finally:
        try:
            pool.close()
            pool.join()
        except Exception:
            pass


if __name__ == "__main__":
    main()
