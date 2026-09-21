from src.capability_inventory import (
    _schema_name_and_description,
    build_capability_inventory,
)


def test_schema_name_and_description():
    schema = {
        "type": "function",
        "function": {
            "name": "example_tool",
            "description": "Example capability.",
            "parameters": {},
        },
    }

    name, description = _schema_name_and_description(schema)

    assert name == "example_tool"
    assert description == "Example capability."


def test_builtin_inventory_contains_known_tools():
    inventory = build_capability_inventory()
    by_name = {item.name: item for item in inventory}

    assert "web_search" in by_name
    assert "web_fetch" in by_name
    assert "manage_settings" in by_name
    assert "manage_mcp" in by_name

    assert by_name["web_search"].source == "builtin"
    assert by_name["web_search"].available is True
    assert by_name["web_search"].requires_connection is False


def test_inventory_does_not_invent_capabilities():
    inventory = build_capability_inventory()
    names = {item.name for item in inventory}

    assert "fake_azure_cost_magic_tool" not in names
    assert "azure_cost_analytics" not in names
    assert "azure_budget_api" not in names


def test_disabled_tool_is_excluded():
    inventory = build_capability_inventory(
        disabled_tools={"web_search"},
    )

    names = {item.name for item in inventory}

    assert "web_search" not in names
    assert "web_fetch" in names


def test_duplicate_mcp_tool_does_not_replace_builtin():
    class FakeMcp:
        def get_all_tools(self):
            return [
                {
                    "name": "web_search",
                    "description": "Fake duplicate.",
                    "server_id": "fake",
                }
            ]

    inventory = build_capability_inventory(
        mcp_manager=FakeMcp(),
    )

    matches = [
        item
        for item in inventory
        if item.name == "web_search"
    ]

    assert len(matches) == 1
    assert matches[0].source == "builtin"


def test_connected_mcp_tool_is_included():
    class FakeMcp:
        def get_all_tools(self):
            return [
                {
                    "qualified_name": "mcp__azure__read_costs",
                    "name": "read_costs",
                    "description": "Read Azure subscription cost data.",
                    "server_id": "azure",
                }
            ]

    inventory = build_capability_inventory(
        mcp_manager=FakeMcp(),
    )

    by_name = {item.name: item for item in inventory}

    assert "mcp__azure__read_costs" in by_name

    capability = by_name["mcp__azure__read_costs"]

    assert capability.source == "mcp"
    assert capability.available is True
    assert capability.metadata["server_id"] == "azure"


from src.capability_inventory import (
    AvailableCapability,
    CapabilityResolutionStatus,
    resolve_capability,
    resolve_capabilities,
)


def test_resolver_finds_strong_web_search_match():
    inventory = [
        AvailableCapability(
            name="web_search",
            description="Search current public web information.",
            source="builtin",
        )
    ]

    result = resolve_capability(
        "Search current public web information",
        inventory,
    )

    assert result.status == CapabilityResolutionStatus.AVAILABLE
    assert result.matched_tool == "web_search"
    assert result.source == "builtin"


def test_resolver_marks_unavailable_azure_cost_capability_missing():
    inventory = [
        AvailableCapability(
            name="web_search",
            description="Search current public web information.",
            source="builtin",
        ),
        AvailableCapability(
            name="read_file",
            description="Read a file from the local workspace.",
            source="builtin",
        ),
    ]

    result = resolve_capability(
        "Read Azure subscription cost and usage data",
        inventory,
    )

    assert result.status == CapabilityResolutionStatus.MISSING
    assert result.matched_tool is None


def test_resolver_finds_connected_azure_mcp_capability():
    inventory = [
        AvailableCapability(
            name="mcp__azure__read_costs",
            description="Read Azure subscription cost and usage data.",
            source="mcp",
            metadata={"server_id": "azure"},
        )
    ]

    result = resolve_capability(
        "Read Azure subscription cost and usage data",
        inventory,
    )

    assert result.status == CapabilityResolutionStatus.AVAILABLE
    assert result.matched_tool == "mcp__azure__read_costs"
    assert result.source == "mcp"


def test_resolver_returns_unknown_for_partial_match():
    inventory = [
        AvailableCapability(
            name="azure_inventory",
            description="Read Azure resource inventory.",
            source="mcp",
        )
    ]

    result = resolve_capability(
        "Read Azure resource inventory and utilization metrics",
        inventory,
    )

    assert result.status == CapabilityResolutionStatus.UNKNOWN
    assert result.matched_tool == "azure_inventory"


def test_resolver_ignores_unavailable_capability():
    inventory = [
        AvailableCapability(
            name="mcp__azure__read_costs",
            description="Read Azure subscription cost and usage data.",
            source="mcp",
            available=False,
            requires_connection=True,
        )
    ]

    result = resolve_capability(
        "Read Azure subscription cost and usage data",
        inventory,
    )

    assert result.status == CapabilityResolutionStatus.MISSING
    assert result.matched_tool is None


def test_resolve_multiple_capabilities_independently():
    inventory = [
        AvailableCapability(
            name="web_search",
            description="Search current public web information.",
            source="builtin",
        ),
        AvailableCapability(
            name="mcp__azure__read_costs",
            description="Read Azure subscription cost and usage data.",
            source="mcp",
        ),
    ]

    results = resolve_capabilities(
        [
            "Search current public web information",
            "Read Azure subscription cost and usage data",
            "Read Azure budget thresholds",
        ],
        inventory,
    )

    assert len(results) == 3
    assert results[0].status == CapabilityResolutionStatus.AVAILABLE
    assert results[1].status == CapabilityResolutionStatus.AVAILABLE
    # Azure cost access is related enough to budget data to warrant review,
    # but it is not strong enough evidence to prove budget access exists.
    assert results[2].status == CapabilityResolutionStatus.UNKNOWN
