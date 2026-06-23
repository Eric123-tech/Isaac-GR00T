#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash scripts/quick_git_push.sh [branch] [commit-message]
  bash scripts/quick_git_push.sh -b <branch> -m <commit-message>

Defaults:
  branch: quick-YYYYmmdd-HHMMSS
  commit-message: 1

Options:
  -b, --branch <name>      Branch to create or switch to.
  -m, --message <message>  Commit message.
  --include-data           Include datasets/ changes. Default excludes datasets/.
  --no-push                Commit locally but do not push.
  --allow-empty            Create an empty commit if there are no staged changes.
  -h, --help               Show this help.

Examples:
  bash scripts/quick_git_push.sh my-branch
  bash scripts/quick_git_push.sh my-branch "1"
  bash scripts/quick_git_push.sh -b gr1-demo -m "Add GR1 demo notes"
EOF
}

branch=""
message="1"
include_data=0
push=1
allow_empty=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -b|--branch)
            branch="${2:-}"
            shift 2
            ;;
        -m|--message)
            message="${2:-}"
            shift 2
            ;;
        --include-data)
            include_data=1
            shift
            ;;
        --no-push)
            push=0
            shift
            ;;
        --allow-empty)
            allow_empty=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            if [[ -z "$branch" ]]; then
                branch="$1"
            elif [[ "$message" == "1" ]]; then
                message="$1"
            else
                echo "Unexpected argument: $1" >&2
                usage >&2
                exit 2
            fi
            shift
            ;;
    esac
done

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

if [[ -z "$branch" ]]; then
    branch="quick-$(date +%Y%m%d-%H%M%S)"
fi

if [[ -z "$message" ]]; then
    message="1"
fi

if ! git remote get-url origin >/dev/null 2>&1; then
    echo "Remote 'origin' is not configured. Add your GitHub repo as origin first." >&2
    exit 1
fi

current_branch="$(git branch --show-current)"
if git show-ref --verify --quiet "refs/heads/$branch"; then
    if [[ "$current_branch" != "$branch" ]]; then
        git checkout "$branch"
    fi
else
    git checkout -b "$branch"
fi

if [[ "$include_data" -eq 1 ]]; then
    git add -A .
else
    git add -u
    mapfile -d "" untracked_paths < <(git ls-files --others --exclude-standard -z)
    if [[ "${#untracked_paths[@]}" -gt 0 ]]; then
        git add -- "${untracked_paths[@]}"
    fi
fi

if git diff --cached --quiet; then
    if [[ "$allow_empty" -eq 1 ]]; then
        git commit --allow-empty -m "$message"
    else
        echo "No staged changes to commit."
        if [[ -n "$(git status --porcelain -- datasets 2>/dev/null)" ]]; then
            echo "Note: datasets/ changes were left unstaged. Use --include-data if you really want to commit them."
        fi
        git status --short
        exit 0
    fi
else
    git commit -m "$message"
fi

if [[ "$push" -eq 1 ]]; then
    git push -u origin "$branch"
else
    echo "Committed locally. Push skipped because --no-push was set."
fi
