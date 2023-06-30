from elk.run import fetch_git_hash


def test_find_git_hash():
    git_hash = fetch_git_hash()
    assert git_hash != "ERROR_HASH_NOT_FOUND"
