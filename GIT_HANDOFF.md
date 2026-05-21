# Git handoff

This repository was built from the uploaded `dataset-artifacts.zip`, including its `.git` directory. The original `origin` remote is preserved:

```text
https://github.com/PaulKMandal/dataset-artifacts
```

The changes are on branch `fast-dynamics-nix-uv` as one commit on top of `origin/main`.

## Inspect and push

```bash
cd dataset-artifacts
git status
git log --oneline --decorate --max-count=5
git show --stat HEAD
git push -u origin fast-dynamics-nix-uv
```

## Author identity

The commit author is set to the author identity used by most existing commits in the uploaded repo:

```text
PaulKMandal <p.mandal@aol.com>
```

If you want GitHub's no-reply address instead, amend before pushing:

```bash
git commit --amend --author='PaulKMandal <EXACT_NOREPLY_FROM_GITHUB_SETTINGS>' --no-edit
git push -u origin fast-dynamics-nix-uv
```

## Patch-only workflow for later iterations

After this environment is merged, subsequent code changes can be handed over as ordinary patches:

```bash
git am 0001-some-change.patch
# or
git apply some-change.patch
```
