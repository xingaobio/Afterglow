import hashlib

from PIL import Image, ImageFilter

from afterglow.util.hashing import dhash, hamming, sha256_file
from tests.conftest import make_sharp


def test_sha256_file_matches_hashlib(tmp_path):
    p = tmp_path / "blob.bin"
    data = b"afterglow" * 300_000
    p.write_bytes(data)
    assert sha256_file(p) == hashlib.sha256(data).hexdigest()
    assert sha256_file(p, chunk=7) == hashlib.sha256(data).hexdigest()


def test_dhash_is_64_bit_hex_and_stable():
    h = dhash(make_sharp(1))
    assert len(h) == 16 and int(h, 16) >= 0
    assert dhash(make_sharp(1)) == h


def test_dhash_distance_small_for_similar_large_for_different():
    base = make_sharp(1)
    blurred = base.filter(ImageFilter.GaussianBlur(2))
    other = make_sharp(99)
    assert hamming(dhash(base), dhash(blurred)) <= 12
    assert hamming(dhash(base), dhash(other)) >= 12


def test_hamming_known_values():
    assert hamming("0000000000000000", "0000000000000000") == 0
    assert hamming("ffffffffffffffff", "0000000000000000") == 64
    assert hamming("0000000000000003", "0000000000000001") == 1


def test_dhash_flat_image_is_zero():
    assert dhash(Image.new("RGB", (50, 50), (10, 10, 10))) == "0" * 16
