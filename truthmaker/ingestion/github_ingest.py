"""GitHub Projects v2 ingestion service for Weaviate.

Fetches Projects, Items, Issues, and PRs from GitHub GraphQL API
and upserts them into Weaviate with proper cross-references.

Usage:
    GITHUB_TOKEN=<token> python truthmaker/ingestion/github_ingest.py

Environment Variables:
    GITHUB_TOKEN: GitHub personal access token with read:project scope
    WEAVIATE_URL: Weaviate endpoint (default: http://localhost:8080)
    GITHUB_ORG: Organization name (default: mhold3n)
    GITHUB_REPO: Repository name (default: Larrak)
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from typing import Any

import weaviate

# GraphQL query to fetch a Project v2 with its items (for organizations)
ORG_PROJECT_QUERY = """
query($owner: String!, $number: Int!) {
  organization(login: $owner) {
    projectV2(number: $number) {
      id
      number
      title
      shortDescription
      url
      items(first: 100) {
        nodes {
          id
          type
          fieldValues(first: 20) {
            nodes {
              ... on ProjectV2ItemFieldTextValue {
                text
                field { ... on ProjectV2FieldCommon { name } }
              }
              ... on ProjectV2ItemFieldSingleSelectValue {
                name
                field { ... on ProjectV2FieldCommon { name } }
              }
            }
          }
          content {
            ... on Issue {
              __typename
              id
              number
              title
              body
              state
              url
              createdAt
              updatedAt
              labels(first: 10) { nodes { name } }
              assignees(first: 5) { nodes { login } }
            }
            ... on PullRequest {
              __typename
              id
              number
              title
              body
              state
              url
              headRefName
              baseRefName
              createdAt
              mergedAt
            }
            ... on DraftIssue {
              __typename
              id
              title
              body
            }
          }
        }
      }
    }
  }
}
"""

# GraphQL query for user-owned projects (personal accounts)
USER_PROJECT_QUERY = """
query($owner: String!, $number: Int!) {
  user(login: $owner) {
    projectV2(number: $number) {
      id
      number
      title
      shortDescription
      url
      items(first: 100) {
        nodes {
          id
          type
          fieldValues(first: 20) {
            nodes {
              ... on ProjectV2ItemFieldTextValue {
                text
                field { ... on ProjectV2FieldCommon { name } }
              }
              ... on ProjectV2ItemFieldSingleSelectValue {
                name
                field { ... on ProjectV2FieldCommon { name } }
              }
            }
          }
          content {
            ... on Issue {
              __typename
              id
              number
              title
              body
              state
              url
              createdAt
              updatedAt
              labels(first: 10) { nodes { name } }
              assignees(first: 5) { nodes { login } }
            }
            ... on PullRequest {
              __typename
              id
              number
              title
              body
              state
              url
              headRefName
              baseRefName
              createdAt
              mergedAt
            }
            ... on DraftIssue {
              __typename
              id
              title
              body
            }
          }
        }
      }
    }
  }
}
"""


def run_graphql_query(query: str, variables: dict[str, Any], token: str) -> dict:
    """Execute a GraphQL query against GitHub API."""
    import urllib.request

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    data = json.dumps({"query": query, "variables": variables}).encode()

    req = urllib.request.Request(
        "https://api.github.com/graphql",
        data=data,
        headers=headers,
        method="POST",
    )

    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())


def parse_iso_date(date_str: str | None) -> datetime | None:
    """Parse ISO 8601 date string to datetime."""
    if not date_str:
        return None
    return datetime.fromisoformat(date_str.replace("Z", "+00:00"))


def upsert_issue(client: weaviate.WeaviateClient, issue_data: dict) -> str:
    """Upsert a GitHub issue into Weaviate. Returns the Weaviate UUID."""
    issues = client.collections.get("GitHubIssue")

    # Check if exists by node_id
    result = issues.query.fetch_objects(
        filters=weaviate.classes.query.Filter.by_property("node_id").equal(issue_data["id"]),
        limit=1,
    )

    properties = {
        "node_id": issue_data["id"],
        "number": issue_data["number"],
        "title": issue_data["title"],
        "body": issue_data.get("body") or "",
        "state": issue_data["state"],
        "labels": [l["name"] for l in issue_data.get("labels", {}).get("nodes", [])],
        "assignees": [a["login"] for a in issue_data.get("assignees", {}).get("nodes", [])],
        "url": issue_data["url"],
        "created_at": parse_iso_date(issue_data.get("createdAt")),
        "updated_at": parse_iso_date(issue_data.get("updatedAt")),
    }

    if result.objects:
        # Update existing
        uuid = result.objects[0].uuid
        issues.data.update(uuid=uuid, properties=properties)
        return str(uuid)
    else:
        # Create new
        uuid = issues.data.insert(properties=properties)
        return str(uuid)


def upsert_pull_request(client: weaviate.WeaviateClient, pr_data: dict) -> str:
    """Upsert a GitHub PR into Weaviate. Returns the Weaviate UUID."""
    prs = client.collections.get("GitHubPullRequest")

    result = prs.query.fetch_objects(
        filters=weaviate.classes.query.Filter.by_property("node_id").equal(pr_data["id"]),
        limit=1,
    )

    properties = {
        "node_id": pr_data["id"],
        "number": pr_data["number"],
        "title": pr_data["title"],
        "body": pr_data.get("body") or "",
        "state": pr_data["state"],
        "head_ref": pr_data.get("headRefName") or "",
        "base_ref": pr_data.get("baseRefName") or "",
        "url": pr_data["url"],
        "created_at": parse_iso_date(pr_data.get("createdAt")),
        "merged_at": parse_iso_date(pr_data.get("mergedAt")),
    }

    if result.objects:
        uuid = result.objects[0].uuid
        prs.data.update(uuid=uuid, properties=properties)
        return str(uuid)
    else:
        uuid = prs.data.insert(properties=properties)
        return str(uuid)


def upsert_draft_issue(client: weaviate.WeaviateClient, draft_data: dict) -> str:
    """Upsert a draft issue into Weaviate. Returns the Weaviate UUID."""
    drafts = client.collections.get("GitHubDraftIssue")

    result = drafts.query.fetch_objects(
        filters=weaviate.classes.query.Filter.by_property("node_id").equal(draft_data["id"]),
        limit=1,
    )

    properties = {
        "node_id": draft_data["id"],
        "title": draft_data["title"],
        "body": draft_data.get("body") or "",
    }

    if result.objects:
        uuid = result.objects[0].uuid
        drafts.data.update(uuid=uuid, properties=properties)
        return str(uuid)
    else:
        uuid = drafts.data.insert(properties=properties)
        return str(uuid)


def upsert_project_item(
    client: weaviate.WeaviateClient,
    item_data: dict,
    content_uuid: str | None,
    content_type: str | None,
) -> str:
    """Upsert a project item into Weaviate."""
    items = client.collections.get("GitHubProjectItem")

    # Extract custom field values as JSON
    field_values = {}
    for fv in item_data.get("fieldValues", {}).get("nodes", []):
        if "field" in fv and fv["field"]:
            name = fv["field"].get("name", "")
            value = fv.get("text") or fv.get("name") or ""
            if name:
                field_values[name] = value

    result = items.query.fetch_objects(
        filters=weaviate.classes.query.Filter.by_property("node_id").equal(item_data["id"]),
        limit=1,
    )

    properties = {
        "node_id": item_data["id"],
        "item_type": item_data.get("type") or "UNKNOWN",
        "status": field_values.get("Status", ""),
        "custom_fields": json.dumps(field_values),
    }

    # Build references based on content type
    references = {}
    if content_uuid and content_type:
        if content_type == "Issue":
            references["content_issue"] = content_uuid
        elif content_type == "PullRequest":
            references["content_pr"] = content_uuid
        elif content_type == "DraftIssue":
            references["content_draft"] = content_uuid

    if result.objects:
        uuid = result.objects[0].uuid
        items.data.update(uuid=uuid, properties=properties, references=references)
        return str(uuid)
    else:
        uuid = items.data.insert(properties=properties, references=references)
        return str(uuid)


def upsert_project(
    client: weaviate.WeaviateClient,
    project_data: dict,
    item_uuids: list[str],
) -> str:
    """Upsert a GitHub Project into Weaviate."""
    projects = client.collections.get("GitHubProject")

    result = projects.query.fetch_objects(
        filters=weaviate.classes.query.Filter.by_property("node_id").equal(project_data["id"]),
        limit=1,
    )

    properties = {
        "node_id": project_data["id"],
        "number": project_data["number"],
        "title": project_data["title"],
        "description": project_data.get("shortDescription") or "",
        "url": project_data["url"],
        "last_synced": datetime.now(timezone.utc),
    }

    references = {"has_items": item_uuids} if item_uuids else {}

    if result.objects:
        uuid = result.objects[0].uuid
        projects.data.update(uuid=uuid, properties=properties, references=references)
        return str(uuid)
    else:
        uuid = projects.data.insert(properties=properties, references=references)
        return str(uuid)


def sync_project(
    client: weaviate.WeaviateClient,
    owner: str,
    project_number: int,
    token: str,
) -> None:
    """Sync a single GitHub Project to Weaviate."""
    print(f"Fetching project {owner}/#{project_number}...")

    # Try user query first (for personal accounts), then organization
    response = run_graphql_query(
        USER_PROJECT_QUERY,
        {"owner": owner, "number": project_number},
        token,
    )

    project = response.get("data", {}).get("user", {}).get("projectV2")

    # If user query failed, try organization query
    if not project:
        response = run_graphql_query(
            ORG_PROJECT_QUERY,
            {"owner": owner, "number": project_number},
            token,
        )
        project = response.get("data", {}).get("organization", {}).get("projectV2")

    if "errors" in response and not project:
        print(f"GraphQL errors: {response['errors']}")
        return

    if not project:
        print(f"Project not found: {owner}/#{project_number}")
        return

    print(f"Processing project: {project['title']}")

    # Process items and their content
    item_uuids = []
    items = project.get("items", {}).get("nodes", [])
    print(f"Found {len(items)} items")

    for item in items:
        content = item.get("content")
        content_uuid = None
        content_type = None

        if content:
            typename = content.get("__typename")
            content_type = typename

            if typename == "Issue":
                content_uuid = upsert_issue(client, content)
                print(f"  Upserted Issue #{content['number']}: {content['title']}")
            elif typename == "PullRequest":
                content_uuid = upsert_pull_request(client, content)
                print(f"  Upserted PR #{content['number']}: {content['title']}")
            elif typename == "DraftIssue":
                content_uuid = upsert_draft_issue(client, content)
                print(f"  Upserted Draft: {content['title']}")

        item_uuid = upsert_project_item(client, item, content_uuid, content_type)
        item_uuids.append(item_uuid)

    # Upsert the project itself with item references
    project_uuid = upsert_project(client, project, item_uuids)
    print(f"Upserted Project: {project['title']} (UUID: {project_uuid})")


def main() -> int:
    """Main entry point."""
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("Error: GITHUB_TOKEN environment variable required")
        return 1

    weaviate_url = os.environ.get("WEAVIATE_URL", "http://localhost:8080")
    org = os.environ.get("GITHUB_ORG", "mhold3n")
    project_numbers_str = os.environ.get("GITHUB_PROJECT_NUMBERS", "1")

    print(f"Connecting to Weaviate at {weaviate_url}...")
    client = weaviate.connect_to_local(port=8080, grpc_port=50052)

    try:
        # Sync each project
        for num_str in project_numbers_str.split(","):
            project_number = int(num_str.strip())
            sync_project(client, org, project_number, token)

        print("Sync complete.")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
