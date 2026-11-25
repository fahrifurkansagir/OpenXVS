"""Pure-Python RFC 8032 Ed25519 signing & verification using Python stdlib `hashlib.sha512` only.

Implements RFC 8032 §5.1 (Ed25519 over twisted Edwards curve -x^2 + y^2 = 1 + d*x^2*y^2 mod 2^255-19)
with zero third-party dependencies, official RFC 8032 §7.1 Test Vectors 1-3, and a bidirectional
WSL2 `cryptography` cross-verification helper (`pure <-> cryptography`).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from typing import Any

# Curve25519 / Edwards25519 constants (RFC 8032 §5.1)
_P: int = (1 << 255) - 19
_L: int = (1 << 252) + 27742317777372353535851937790883648493
_D: int = (-121665 * pow(121666, _P - 2, _P)) % _P
_I: int = pow(2, (_P - 1) // 4, _P)  # sqrt(-1) mod p

# Official RFC 8032 §7.1 Test Vectors 1, 2, 3 (empty, 1-byte 0x72, 2-byte 0xaf82)
RFC8032_TEST_VECTORS: tuple[dict[str, str], ...] = (
    {
        "name": "RFC8032_7_1_TEST1",
        "secret_hex": "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60",
        "public_hex": "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a",
        "message_hex": "",
        "signature_hex": (
            "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e06522490155"
            "5fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b"
        ),
    },
    {
        "name": "RFC8032_7_1_TEST2",
        "secret_hex": "4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb",
        "public_hex": "3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c",
        "message_hex": "72",
        "signature_hex": (
            "92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da"
            "085ac1e43e15996e458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00"
        ),
    },
    {
        "name": "RFC8032_7_1_TEST3",
        "secret_hex": "c5aa8df43f9f837bedb7442f31dcb7b166d38535076f094b85ce3a2e0b4458f7",
        "public_hex": "fc51cd8e6218a1a38da47ed00230f0580816ed13ba3303ac5deb911548908025",
        "message_hex": "af82",
        "signature_hex": (
            "6291d657deec24024827e69c3abe01a30ce548a284743a445e3680d7db5ac3ac"
            "18ff9b538d16f290ae67f760984dc6594a7c15e9716ed28dc027beceea1ec40a"
        ),
    },
)


def _sha512(data: bytes) -> bytes:
    return hashlib.sha512(data).digest()


def _recover_x(y: int, sign: int) -> int | None:
    """Recover x coordinate on Edwards25519 from y and sign bit (RFC 8032 §5.1.3)."""
    if y >= _P:
        return None
    x2 = ((y * y - 1) * pow(_D * y * y + 1, _P - 2, _P)) % _P
    if x2 == 0:
        return None if sign == 1 else 0
    x = pow(x2, (_P + 3) // 8, _P)
    if (x * x - x2) % _P != 0:
        x = (x * _I) % _P
    if (x * x - x2) % _P != 0:
        return None
    if (x & 1) != sign:
        x = _P - x
    return x


# Base point B in extended coordinates (X, Y, Z, T)
_BY: int = (4 * pow(5, _P - 2, _P)) % _P
_BX: int = _recover_x(_BY, 0) or 0
_BASE_POINT: tuple[int, int, int, int] = (_BX, _BY, 1, (_BX * _BY) % _P)
_IDENTITY: tuple[int, int, int, int] = (0, 1, 1, 0)


def _point_add(
    p1: tuple[int, int, int, int],
    p2: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Complete unified addition on Edwards25519 in extended coordinates (RFC 8032 §5.1.4)."""
    x1, y1, z1, t1 = p1
    x2, y2, z2, t2 = p2
    a = ((y1 - x1) * (y2 - x2)) % _P
    b = ((y1 + x1) * (y2 + x2)) % _P
    c = (2 * t1 * t2 * _D) % _P
    d = (2 * z1 * z2) % _P
    e = (b - a) % _P
    f = (d - c) % _P
    g = (d + c) % _P
    h = (b + a) % _P
    return ((e * f) % _P, (g * h) % _P, (f * g) % _P, (e * h) % _P)


def _point_mul(scalar: int, point: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Double-and-add scalar multiplication on Edwards25519."""
    res = _IDENTITY
    addend = point
    k = scalar
    while k > 0:
        if k & 1:
            res = _point_add(res, addend)
        addend = _point_add(addend, addend)
        k >>= 1
    return res


def _point_equal(
    p1: tuple[int, int, int, int],
    p2: tuple[int, int, int, int],
) -> bool:
    x1, y1, z1, _ = p1
    x2, y2, z2, _ = p2
    return ((x1 * z2 - x2 * z1) % _P == 0) and ((y1 * z2 - y2 * z1) % _P == 0)


def _encode_point(pt: tuple[int, int, int, int]) -> bytes:
    x, y, z, _ = pt
    zi = pow(z, _P - 2, _P)
    xa = (x * zi) % _P
    ya = (y * zi) % _P
    encoded = ya | ((xa & 1) << 255)
    return encoded.to_bytes(32, "little")


def _decode_point(raw: bytes) -> tuple[int, int, int, int] | None:
    if len(raw) != 32:
        return None
    val = int.from_bytes(raw, "little")
    y = val & ((1 << 255) - 1)
    sign = (val >> 255) & 1
    x = _recover_x(y, sign)
    if x is None:
        return None
    return (x, y, 1, (x * y) % _P)


def _expand_seed(seed32: bytes) -> tuple[int, bytes]:
    if len(seed32) != 32:
        raise ValueError(f"Ed25519 private seed must be exactly 32 bytes, got {len(seed32)}")
    h = bytearray(_sha512(seed32))
    h[0] &= 248
    h[31] &= 127
    h[31] |= 64
    scalar = int.from_bytes(bytes(h[:32]), "little")
    prefix = bytes(h[32:])
    return scalar, prefix


def ed25519_public_key(seed32: bytes) -> bytes:
    """Derive 32-byte RFC 8032 Ed25519 public key from 32-byte private seed."""
    scalar, _ = _expand_seed(seed32)
    return _encode_point(_point_mul(scalar, _BASE_POINT))


def ed25519_sign(seed32: bytes, message: bytes) -> bytes:
    """Produce deterministic 64-byte RFC 8032 Ed25519 signature over `message`."""
    scalar, prefix = _expand_seed(seed32)
    pub = _encode_point(_point_mul(scalar, _BASE_POINT))
    r = int.from_bytes(_sha512(prefix + message), "little") % _L
    r_enc = _encode_point(_point_mul(r, _BASE_POINT))
    k = int.from_bytes(_sha512(r_enc + pub + message), "little") % _L
    s = (r + k * scalar) % _L
    return r_enc + s.to_bytes(32, "little")


def ed25519_verify(public_key32: bytes, message: bytes, signature64: bytes) -> bool:
    """Verify 64-byte RFC 8032 Ed25519 signature against 32-byte public key (fail-closed)."""
    if len(public_key32) != 32 or len(signature64) != 64:
        return False
    a_pt = _decode_point(public_key32)
    r_pt = _decode_point(signature64[:32])
    if a_pt is None or r_pt is None:
        return False
    s = int.from_bytes(signature64[32:], "little")
    if s >= _L:
        return False
    k = int.from_bytes(_sha512(signature64[:32] + public_key32 + message), "little") % _L
    lhs = _point_mul(s, _BASE_POINT)
    rhs = _point_add(r_pt, _point_mul(k, a_pt))
    return _point_equal(lhs, rhs)


def verify_rfc8032_test_vectors() -> dict[str, Any]:
    """Verify pure-Python Ed25519 against official RFC 8032 §7.1 Test Vectors 1, 2, and 3."""
    passed = 0
    results: list[dict[str, Any]] = []
    for vec in RFC8032_TEST_VECTORS:
        seed = bytes.fromhex(vec["secret_hex"])
        expected_pk = bytes.fromhex(vec["public_hex"])
        msg = bytes.fromhex(vec["message_hex"])
        expected_sig = bytes.fromhex(vec["signature_hex"])

        pk = ed25519_public_key(seed)
        sig = ed25519_sign(seed, msg)
        ok_verify = ed25519_verify(pk, msg, sig)
        bad_sig = bytearray(sig)
        bad_sig[0] ^= 0x01
        ok_reject_tamper = not ed25519_verify(pk, msg, bytes(bad_sig))

        ok = (pk == expected_pk) and (sig == expected_sig) and ok_verify and ok_reject_tamper
        if ok:
            passed += 1
        results.append(
            {
                "name": vec["name"],
                "pubkey_match": pk == expected_pk,
                "sig_match": sig == expected_sig,
                "verify_ok": ok_verify,
                "tamper_rejected": ok_reject_tamper,
                "pass": ok,
            }
        )
    return {
        "vectors_passed": passed,
        "vectors_total": len(RFC8032_TEST_VECTORS),
        "all_pass": passed == len(RFC8032_TEST_VECTORS),
        "details": results,
    }


def cross_verify_with_wsl2_cryptography(seed32: bytes, message: bytes) -> dict[str, Any]:
    """Bidirectionally cross-verify pure-Python Ed25519 against WSL2 `cryptography` Ed25519."""
    pk_pure = ed25519_public_key(seed32)
    sig_pure = ed25519_sign(seed32, message)

    wsl_code = (
        "import json, sys, cryptography; "
        "from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey; "
        "from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat; "
        "payload = json.loads(sys.stdin.read()); "
        "seed = bytes.fromhex(payload['seed_hex']); "
        "msg = bytes.fromhex(payload['msg_hex']); "
        "pk_pure = bytes.fromhex(payload['pk_hex']); "
        "sig_pure = bytes.fromhex(payload['sig_hex']); "
        "priv = Ed25519PrivateKey.from_private_bytes(seed); "
        "pk_crypto = priv.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw); "
        "Ed25519PublicKey.from_public_bytes(pk_pure).verify(sig_pure, msg); "
        "sig_crypto = priv.sign(msg); "
        "print(json.dumps({'cryptography_version': cryptography.__version__, "
        "'pk_match': pk_crypto == pk_pure, 'sig_crypto_hex': sig_crypto.hex(), "
        "'pure_sig_verified_by_crypto': True}))"
    )
    req = json.dumps(
        {
            "seed_hex": seed32.hex(),
            "msg_hex": message.hex(),
            "pk_hex": pk_pure.hex(),
            "sig_hex": sig_pure.hex(),
        }
    )
    proc = subprocess.run(
        ["wsl", "-d", "Ubuntu", "--", "python3", "-c", wsl_code],
        input=req,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    res = json.loads(proc.stdout.strip())
    sig_crypto = bytes.fromhex(res["sig_crypto_hex"])
    crypto_sig_verified_by_pure = ed25519_verify(pk_pure, message, sig_crypto)
    exact_sig_match = sig_crypto == sig_pure
    return {
        "cryptography_version": str(res["cryptography_version"]),
        "pk_match": bool(res["pk_match"]),
        "pure_sig_verified_by_crypto": bool(res["pure_sig_verified_by_crypto"]),
        "crypto_sig_verified_by_pure": bool(crypto_sig_verified_by_pure),
        "bit_exact_signature_match": bool(exact_sig_match),
        "pass": bool(
            res["pk_match"]
            and res["pure_sig_verified_by_crypto"]
            and crypto_sig_verified_by_pure
            and exact_sig_match
        ),
    }
