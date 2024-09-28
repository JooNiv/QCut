#! /bin/bash

# Function to compare two versions
function version_gt() {
  # Remove the 'v' prefix for comparison
  local ver1="${1#v}"
  local ver2="${2#v}"

  if [ "$ver1" = "$ver2" ]; then
    return 0
  elif [ "$(printf '%s\n' "$ver1" "$ver2" | sort -V | head -n 1)" != "$ver1" ]; then
    return 0
  else
    return 1
  fi
}

# Function to extract the version from the changelog
function get_version_in_changelog() {
  for i in 1 2 3 4 5 6 7
  do
    version_line=$(sed "${i}q;d" CHANGELOG.rst) # Get ith line of file
    set -- $version_line
    version=$2
    if [[ $version =~ ^v?[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
      echo $version $3
      return
    fi
  done
  if [[ ! $version ]]
  then
    printf "\033[0;31mChangelog file is incorrect, one of the first seven lines should be of the format 'Version xx.xx.x' or 'vX.X.X'.\033[0m";
    return 171
  fi
}

# Function to verify if the changelog version is valid
function verify_changelog_version() {
  # Read the version and date from the changelog
  read -r version date < <(get_version_in_changelog)

  # Get the latest Git tag in 'vX.X.X' format
  current_version=$(git tag -l --sort=-version:refname | grep -E '^v?[0-9]+\.[0-9]+\.[0-9]+$' | head -n 1)

  # Check if the new version is greater than the current version
  if version_gt "$current_version" "$version"; then
    printf "\033[0;31mNew version in the changelog (%s) should be greater than the current version (%s).\n\033[0m" "$version" "$current_version";
    return 172
  fi

  # Check if the version already exists as a tag
  if git tag -l | grep -q "^v$version$"; then
    printf "Version %s already exists.\n" "$version";
    return 172
  fi

  printf "Current version is %s, new version is v%s.\n" "$current_version" "$version";
}

# Function to create a new Git tag
function create_new_tag() {
  read -r version date < <(get_version_in_changelog)
  if [ -n "$date" ]; then
    printf "\033[0;33mWarning: content found after the version number, not releasing a new version.\n\033[0m";
    return 173
  fi

  # Ensure the version tag has a 'v' prefix
  if [[ ! "$version" =~ ^v ]]; then
    version="v$version"
  fi

  # Only tag if the version doesn't exist yet
  if ! git tag -l | grep -q "^$version$"; then
    printf "Releasing version %s.\n" "$version"
    curl -X POST -H "Authorization: token $GITHUB_TOKEN" "https://api.github.com/repos/$GITHUB_REPOSITORY/releases" \
         -d "{\"tag_name\": \"$version\", \"name\": \"$version\", \"body\": \"Changelog: https://github.com/$GITHUB_REPOSITORY/blob/main/CHANGELOG.rst\"}"
  fi
}

"$@"
