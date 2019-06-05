#!/bin/bash -e

# A script which migrates all team directories to a new version and creates git branches
# for each of them and pushes them to GitLab. One can then use those branches to create
# merge requests to validate and merge them to a new version of the core package.
# It migrates only directories for those teams which do not yet have a directory in the
# target version's directory.

FROM_VERSION="$1"
TO_VERSION="$2"

if [ -z "$FROM_VERSION" ] || [ -z "$TO_VERSION" ]; then
    echo "Usage: ./migrate.sh <FROM_VERSION> <TO_VERSION>"
    exit 1
fi

git checkout master

if [[ ! -e "$FROM_VERSION" ]]; then
    echo "'$FROM_VERSION' does not exist"
    exit 2
fi

echo ">>> Migrating '$FROM_VERSION' to '$TO_VERSION'"

for FROM_TEAM in $FROM_VERSION/*; do
    TEAM_NAME=$(basename "$FROM_TEAM")
    TO_TEAM="$TO_VERSION/$TEAM_NAME"
    echo ">>> Migrating '$FROM_TEAM' to '$TO_TEAM'"

    if [[ -e "$TO_TEAM" ]]; then
        echo ">>> '$TO_TEAM' already exists, skipping"
        continue
    fi

    FROM_INTERFACE="${FROM_VERSION#v}"
    TO_INTERFACE="${TO_VERSION#v}"

    # TODO: Clean team name better to assure it is a valid branch name.
    BRANCH_NAME="migrate/${TEAM_NAME// /}"

    git branch -D "$BRANCH_NAME" || true
    git checkout -b "$BRANCH_NAME"
    mkdir -p "$TO_VERSION"
    cp -a "$FROM_TEAM" "$TO_TEAM"

    # Migrate primitive annotations themselves. This is calling a helper Python script.
    find "$TO_TEAM" -name primitive.json -print0 | xargs -0 ./migrate.py "$TO_INTERFACE"

    git add "$TO_TEAM"
    git commit -m "Migrating $TEAM_NAME primitives to $TO_VERSION." "$TO_TEAM"
    git push -f origin "$BRANCH_NAME"
    git checkout master
    git branch -D "$BRANCH_NAME"
done
