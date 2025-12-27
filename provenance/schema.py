"""Weaviate schema definition for Larrak provenance system.

Collections are created in dependency order to avoid forward references:
1. Module (no refs)
2. Artifact (refs: Run - added later)
3. Run (refs: Module, Artifact)
4. CodeSymbol (refs: Module)
5. Tracker (refs: CodeSymbol, Artifact)
6. EngineeringElement (refs: Module, Tracker)
7. ComplianceCheck (refs: EngineeringElement)
"""

import weaviate
import weaviate.classes.config as wvc


def create_schema(client: weaviate.WeaviateClient) -> None:
    """Create all Weaviate collections in dependency order.

    Args:
        client: Connected Weaviate client
    """
    # 1. Module (no dependencies)
    if not client.collections.exists("Module"):
        client.collections.create(
            name="Module",
            description="A declared processing module or script",
            properties=[
                wvc.Property(
                    name="module_id", data_type=wvc.DataType.TEXT, skip_vectorization=True
                ),
                wvc.Property(
                    name="entrypoint", data_type=wvc.DataType.TEXT, skip_vectorization=True
                ),
                wvc.Property(name="description", data_type=wvc.DataType.TEXT),
                wvc.Property(name="owner", data_type=wvc.DataType.TEXT),
            ],
            vectorizer_config=wvc.Configure.Vectorizer.none(),
        )

    # 2. Artifact (created before Run to avoid forward reference)
    if not client.collections.exists("Artifact"):
        client.collections.create(
            name="Artifact",
            description="A file or object produced or used by a run",
            properties=[
                wvc.Property(
                    name="artifact_id", data_type=wvc.DataType.TEXT, skip_vectorization=True
                ),
                wvc.Property(name="path", data_type=wvc.DataType.TEXT),
                wvc.Property(name="role", data_type=wvc.DataType.TEXT),
                wvc.Property(name="content_hash", data_type=wvc.DataType.TEXT),
                wvc.Property(name="meta", data_type=wvc.DataType.TEXT),
                wvc.Property(name="summary", data_type=wvc.DataType.TEXT),
            ],
            # generated_by ref to Run added after Run is created
            vectorizer_config=wvc.Configure.Vectorizer.none(),
        )

    # 3. Run (refs: Module, Artifact - both exist now)
    if not client.collections.exists("Run"):
        client.collections.create(
            name="Run",
            description="An execution of a module",
            properties=[
                wvc.Property(name="run_id", data_type=wvc.DataType.TEXT, skip_vectorization=True),
                wvc.Property(name="start_time", data_type=wvc.DataType.DATE),
                wvc.Property(name="end_time", data_type=wvc.DataType.DATE),
                wvc.Property(name="status", data_type=wvc.DataType.TEXT),
                wvc.Property(name="args", data_type=wvc.DataType.TEXT_ARRAY),
                wvc.Property(name="env", data_type=wvc.DataType.TEXT),
                wvc.Property(name="tags", data_type=wvc.DataType.TEXT_ARRAY),
            ],
            references=[
                wvc.ReferenceProperty(name="executed_module", target_collection="Module"),
                wvc.ReferenceProperty(name="used_artifacts", target_collection="Artifact"),
                wvc.ReferenceProperty(name="generated_artifacts", target_collection="Artifact"),
            ],
            vectorizer_config=wvc.Configure.Vectorizer.none(),
        )

    # 3b. Add generated_by ref from Artifact to Run (now that Run exists)
    artifacts = client.collections.get("Artifact")
    current_refs = [r.name for r in artifacts.config.get().references]
    if "generated_by" not in current_refs:
        artifacts.config.add_reference(
            wvc.ReferenceProperty(name="generated_by", target_collection="Run")
        )

    # 4. CodeSymbol (refs: Module)
    if not client.collections.exists("CodeSymbol"):
        client.collections.create(
            name="CodeSymbol",
            description="A function, class, or variable definition",
            properties=[
                wvc.Property(name="name", data_type=wvc.DataType.TEXT),
                wvc.Property(name="file_path", data_type=wvc.DataType.TEXT),
                wvc.Property(name="line_number", data_type=wvc.DataType.INT),
                wvc.Property(name="signature", data_type=wvc.DataType.TEXT),
                wvc.Property(name="docstring", data_type=wvc.DataType.TEXT),
                wvc.Property(name="code_content", data_type=wvc.DataType.TEXT),
            ],
            references=[
                wvc.ReferenceProperty(name="defined_in", target_collection="Module"),
            ],
            vectorizer_config=wvc.Configure.Vectorizer.none(),
        )

    # 5. Tracker (refs: CodeSymbol, Artifact)
    if not client.collections.exists("Tracker"):
        client.collections.create(
            name="Tracker",
            description="A requirement, task, or decision note",
            properties=[
                wvc.Property(
                    name="tracker_id", data_type=wvc.DataType.TEXT, skip_vectorization=True
                ),
                wvc.Property(name="title", data_type=wvc.DataType.TEXT),
                wvc.Property(name="body", data_type=wvc.DataType.TEXT),
                wvc.Property(name="kind", data_type=wvc.DataType.TEXT),
            ],
            references=[
                wvc.ReferenceProperty(name="related_code", target_collection="CodeSymbol"),
                wvc.ReferenceProperty(name="related_artifacts", target_collection="Artifact"),
            ],
            vectorizer_config=wvc.Configure.Vectorizer.none(),
        )

    # 6. EngineeringElement (refs: Module, Tracker)
    if not client.collections.exists("EngineeringElement"):
        client.collections.create(
            name="EngineeringElement",
            description="Tracked engineering element with compliance metadata",
            properties=[
                wvc.Property(
                    name="element_id",
                    data_type=wvc.DataType.TEXT,
                    skip_vectorization=True,
                ),
                wvc.Property(name="element_type", data_type=wvc.DataType.TEXT),
                wvc.Property(name="file_path", data_type=wvc.DataType.TEXT),
                wvc.Property(name="line_range", data_type=wvc.DataType.TEXT),
                wvc.Property(name="name", data_type=wvc.DataType.TEXT),
                wvc.Property(name="unit", data_type=wvc.DataType.TEXT),
                wvc.Property(name="source_citation", data_type=wvc.DataType.TEXT),
                wvc.Property(name="uncertainty", data_type=wvc.DataType.NUMBER),
                wvc.Property(name="valid_range", data_type=wvc.DataType.TEXT),
                wvc.Property(name="compliance_status", data_type=wvc.DataType.TEXT),
                wvc.Property(name="last_verified", data_type=wvc.DataType.DATE),
                wvc.Property(name="docstring", data_type=wvc.DataType.TEXT),
            ],
            references=[
                wvc.ReferenceProperty(name="in_module", target_collection="Module"),
                wvc.ReferenceProperty(name="related_requirements", target_collection="Tracker"),
            ],
            vectorizer_config=wvc.Configure.Vectorizer.none(),
        )

    # 7. ComplianceCheck (refs: EngineeringElement)
    if not client.collections.exists("ComplianceCheck"):
        client.collections.create(
            name="ComplianceCheck",
            description="Audit trail entry for engineering compliance verification",
            properties=[
                wvc.Property(
                    name="check_id",
                    data_type=wvc.DataType.TEXT,
                    skip_vectorization=True,
                ),
                wvc.Property(name="check_type", data_type=wvc.DataType.TEXT),
                wvc.Property(name="status", data_type=wvc.DataType.TEXT),
                wvc.Property(name="message", data_type=wvc.DataType.TEXT),
                wvc.Property(name="checked_at", data_type=wvc.DataType.DATE),
            ],
            references=[
                wvc.ReferenceProperty(name="element", target_collection="EngineeringElement"),
            ],
            vectorizer_config=wvc.Configure.Vectorizer.none(),
        )

    # 8. Geometry (refs: Run)
    if not client.collections.exists("Geometry"):
        client.collections.create(
            name="Geometry",
            description="A geometric object produced by the ShapeKernel",
            properties=[
                wvc.Property(
                    name="geometry_id", data_type=wvc.DataType.TEXT, skip_vectorization=True
                ),
                wvc.Property(name="voxel_file_path", data_type=wvc.DataType.TEXT),
                wvc.Property(name="mesh_file_path", data_type=wvc.DataType.TEXT),
                wvc.Property(name="volume_mm3", data_type=wvc.DataType.NUMBER),
                wvc.Property(name="surface_area_mm2", data_type=wvc.DataType.NUMBER),
                wvc.Property(name="mesh_hash", data_type=wvc.DataType.TEXT),
                wvc.Property(name="created_at", data_type=wvc.DataType.DATE),
            ],
            references=[
                wvc.ReferenceProperty(name="generated_by", target_collection="Run"),
            ],
            vectorizer_config=wvc.Configure.Vectorizer.none(),
        )

    # 9. ConstraintResult (refs: Geometry)
    if not client.collections.exists("ConstraintResult"):
        client.collections.create(
            name="ConstraintResult",
            description="Outcome of a geometric constraint check",
            properties=[
                wvc.Property(
                    name="result_id", data_type=wvc.DataType.TEXT, skip_vectorization=True
                ),
                wvc.Property(name="check_name", data_type=wvc.DataType.TEXT),
                wvc.Property(name="passed", data_type=wvc.DataType.BOOL),
                wvc.Property(name="margin_mm", data_type=wvc.DataType.NUMBER),
                wvc.Property(name="location_vector", data_type=wvc.DataType.NUMBER_ARRAY),
                wvc.Property(name="message", data_type=wvc.DataType.TEXT),
            ],
            references=[
                wvc.ReferenceProperty(name="checked_on", target_collection="Geometry"),
            ],
            vectorizer_config=wvc.Configure.Vectorizer.none(),
        )

    # ==========================================================================
    # GitHub Integration Collections (Phase 4)
    # ==========================================================================

    # 10. Repository (GitHub repo - no refs initially)
    if not client.collections.exists("Repository"):
        client.collections.create(
            name="Repository",
            description="A GitHub repository",
            properties=[
                wvc.Property(name="node_id", data_type=wvc.DataType.TEXT, skip_vectorization=True),
                wvc.Property(name="name", data_type=wvc.DataType.TEXT),
                wvc.Property(name="owner", data_type=wvc.DataType.TEXT),
                wvc.Property(name="url", data_type=wvc.DataType.TEXT),
                wvc.Property(name="default_branch", data_type=wvc.DataType.TEXT),
                wvc.Property(name="last_indexed", data_type=wvc.DataType.DATE),
            ],
            vectorizer_config=wvc.Configure.Vectorizer.none(),
        )

    # 11. CodeFile (refs: Repository)
    if not client.collections.exists("CodeFile"):
        client.collections.create(
            name="CodeFile",
            description="A file in a repository",
            properties=[
                wvc.Property(name="path", data_type=wvc.DataType.TEXT),
                wvc.Property(name="language", data_type=wvc.DataType.TEXT),
                wvc.Property(name="size_bytes", data_type=wvc.DataType.INT),
                wvc.Property(name="last_indexed", data_type=wvc.DataType.DATE),
            ],
            references=[
                wvc.ReferenceProperty(name="in_repo", target_collection="Repository"),
            ],
            vectorizer_config=wvc.Configure.Vectorizer.none(),
        )

    # 11b. Add defined_in_file ref to CodeSymbol (now that CodeFile exists)
    code_symbols = client.collections.get("CodeSymbol")
    current_refs = [r.name for r in code_symbols.config.get().references]
    if "defined_in_file" not in current_refs:
        code_symbols.config.add_reference(
            wvc.ReferenceProperty(name="defined_in_file", target_collection="CodeFile")
        )

    # 12. GitHubIssue (refs: none initially, Symbol ref added later)
    if not client.collections.exists("GitHubIssue"):
        client.collections.create(
            name="GitHubIssue",
            description="A GitHub issue",
            properties=[
                wvc.Property(name="node_id", data_type=wvc.DataType.TEXT, skip_vectorization=True),
                wvc.Property(name="number", data_type=wvc.DataType.INT),
                wvc.Property(name="title", data_type=wvc.DataType.TEXT),
                wvc.Property(name="body", data_type=wvc.DataType.TEXT),
                wvc.Property(name="state", data_type=wvc.DataType.TEXT),
                wvc.Property(name="labels", data_type=wvc.DataType.TEXT_ARRAY),
                wvc.Property(name="assignees", data_type=wvc.DataType.TEXT_ARRAY),
                wvc.Property(name="url", data_type=wvc.DataType.TEXT),
                wvc.Property(name="created_at", data_type=wvc.DataType.DATE),
                wvc.Property(name="updated_at", data_type=wvc.DataType.DATE),
                # Custom field for confirmed symbol links (from GitHub Projects)
                wvc.Property(name="linked_symbols_field", data_type=wvc.DataType.TEXT),
            ],
            references=[
                wvc.ReferenceProperty(name="relates_to", target_collection="CodeSymbol"),
            ],
            vectorizer_config=wvc.Configure.Vectorizer.none(),
        )

    # 13. GitHubPullRequest (refs: none initially)
    if not client.collections.exists("GitHubPullRequest"):
        client.collections.create(
            name="GitHubPullRequest",
            description="A GitHub pull request",
            properties=[
                wvc.Property(name="node_id", data_type=wvc.DataType.TEXT, skip_vectorization=True),
                wvc.Property(name="number", data_type=wvc.DataType.INT),
                wvc.Property(name="title", data_type=wvc.DataType.TEXT),
                wvc.Property(name="body", data_type=wvc.DataType.TEXT),
                wvc.Property(name="state", data_type=wvc.DataType.TEXT),
                wvc.Property(name="head_ref", data_type=wvc.DataType.TEXT),
                wvc.Property(name="base_ref", data_type=wvc.DataType.TEXT),
                wvc.Property(name="url", data_type=wvc.DataType.TEXT),
                wvc.Property(name="created_at", data_type=wvc.DataType.DATE),
                wvc.Property(name="merged_at", data_type=wvc.DataType.DATE),
            ],
            references=[
                wvc.ReferenceProperty(name="touches_files", target_collection="CodeFile"),
            ],
            vectorizer_config=wvc.Configure.Vectorizer.none(),
        )

    # 14. GitHubDraftIssue
    if not client.collections.exists("GitHubDraftIssue"):
        client.collections.create(
            name="GitHubDraftIssue",
            description="A draft issue in a GitHub Project",
            properties=[
                wvc.Property(name="node_id", data_type=wvc.DataType.TEXT, skip_vectorization=True),
                wvc.Property(name="title", data_type=wvc.DataType.TEXT),
                wvc.Property(name="body", data_type=wvc.DataType.TEXT),
            ],
            vectorizer_config=wvc.Configure.Vectorizer.none(),
        )

    # 15. GitHubProjectItem (refs: Issue/PR/Draft)
    if not client.collections.exists("GitHubProjectItem"):
        client.collections.create(
            name="GitHubProjectItem",
            description="An item in a GitHub Project v2",
            properties=[
                wvc.Property(name="node_id", data_type=wvc.DataType.TEXT, skip_vectorization=True),
                wvc.Property(name="item_type", data_type=wvc.DataType.TEXT),
                wvc.Property(name="status", data_type=wvc.DataType.TEXT),
                wvc.Property(name="custom_fields", data_type=wvc.DataType.TEXT),
            ],
            references=[
                wvc.ReferenceProperty(name="content_issue", target_collection="GitHubIssue"),
                wvc.ReferenceProperty(name="content_pr", target_collection="GitHubPullRequest"),
                wvc.ReferenceProperty(name="content_draft", target_collection="GitHubDraftIssue"),
            ],
            vectorizer_config=wvc.Configure.Vectorizer.none(),
        )

    # 16. GitHubProject (refs: ProjectItem[])
    if not client.collections.exists("GitHubProject"):
        client.collections.create(
            name="GitHubProject",
            description="A GitHub Project v2",
            properties=[
                wvc.Property(name="node_id", data_type=wvc.DataType.TEXT, skip_vectorization=True),
                wvc.Property(name="number", data_type=wvc.DataType.INT),
                wvc.Property(name="title", data_type=wvc.DataType.TEXT),
                wvc.Property(name="description", data_type=wvc.DataType.TEXT),
                wvc.Property(name="url", data_type=wvc.DataType.TEXT),
                wvc.Property(name="last_synced", data_type=wvc.DataType.DATE),
            ],
            references=[
                wvc.ReferenceProperty(name="has_items", target_collection="GitHubProjectItem"),
            ],
            vectorizer_config=wvc.Configure.Vectorizer.none(),
        )

    # 17. WorkflowRun (optional - GitHub Actions linkage)
    if not client.collections.exists("WorkflowRun"):
        client.collections.create(
            name="WorkflowRun",
            description="A GitHub Actions workflow run",
            properties=[
                wvc.Property(name="run_id", data_type=wvc.DataType.TEXT, skip_vectorization=True),
                wvc.Property(name="workflow_name", data_type=wvc.DataType.TEXT),
                wvc.Property(name="status", data_type=wvc.DataType.TEXT),
                wvc.Property(name="conclusion", data_type=wvc.DataType.TEXT),
                wvc.Property(name="run_number", data_type=wvc.DataType.INT),
                wvc.Property(name="started_at", data_type=wvc.DataType.DATE),
                wvc.Property(name="completed_at", data_type=wvc.DataType.DATE),
                wvc.Property(name="url", data_type=wvc.DataType.TEXT),
            ],
            references=[
                wvc.ReferenceProperty(name="touches_files", target_collection="CodeFile"),
                wvc.ReferenceProperty(name="touches_issues", target_collection="GitHubIssue"),
            ],
            vectorizer_config=wvc.Configure.Vectorizer.none(),
        )

    # ==========================================================================
    # ML Model Tracking (HiFi Surrogates)
    # ==========================================================================

    # 18. TrainedModel (tracks expensive ML surrogate models)
    if not client.collections.exists("TrainedModel"):
        client.collections.create(
            name="TrainedModel",
            description="A trained ML surrogate model used in the optimization pipeline",
            properties=[
                wvc.Property(name="model_id", data_type=wvc.DataType.TEXT, skip_vectorization=True),
                wvc.Property(
                    name="model_type", data_type=wvc.DataType.TEXT
                ),  # thermal, structural, flow
                wvc.Property(name="file_path", data_type=wvc.DataType.TEXT),
                wvc.Property(name="training_date", data_type=wvc.DataType.DATE),
                wvc.Property(name="n_samples", data_type=wvc.DataType.INT),
                wvc.Property(name="n_ensemble_members", data_type=wvc.DataType.INT),
                wvc.Property(name="final_loss", data_type=wvc.DataType.NUMBER),
                wvc.Property(name="workflow_stage", data_type=wvc.DataType.TEXT),  # surrogate
                wvc.Property(name="metadata", data_type=wvc.DataType.TEXT),  # JSON hyperparams
            ],
            references=[
                wvc.ReferenceProperty(name="trained_by", target_collection="Run"),
            ],
            vectorizer_config=wvc.Configure.Vectorizer.none(),
        )

    # ==========================================================================
    # HiFi Simulation Runs (Expensive OpenFOAM/CalculiX)
    # ==========================================================================

    # 19. HiFiSimulationRun (tracks expensive CFD/FEA runs at Stage 4: Simulation)
    if not client.collections.exists("HiFiSimulationRun"):
        client.collections.create(
            name="HiFiSimulationRun",
            description="An expensive OpenFOAM or CalculiX simulation run (Stage 4: Ground Truth)",
            properties=[
                wvc.Property(name="run_id", data_type=wvc.DataType.TEXT, skip_vectorization=True),
                wvc.Property(name="solver_type", data_type=wvc.DataType.TEXT),  # openfoam, calculix
                wvc.Property(name="solver_name", data_type=wvc.DataType.TEXT),  # laplacianFoam, ccx
                wvc.Property(name="case_path", data_type=wvc.DataType.TEXT),
                wvc.Property(name="started_at", data_type=wvc.DataType.DATE),
                wvc.Property(name="completed_at", data_type=wvc.DataType.DATE),
                wvc.Property(name="success", data_type=wvc.DataType.BOOL),
                wvc.Property(name="exit_code", data_type=wvc.DataType.INT),
                wvc.Property(name="compute_time_s", data_type=wvc.DataType.NUMBER),
                wvc.Property(name="workflow_stage", data_type=wvc.DataType.TEXT),  # simulation
                # Input parameters
                wvc.Property(name="bore_mm", data_type=wvc.DataType.NUMBER),
                wvc.Property(name="stroke_mm", data_type=wvc.DataType.NUMBER),
                wvc.Property(name="compression_ratio", data_type=wvc.DataType.NUMBER),
                wvc.Property(name="rpm", data_type=wvc.DataType.NUMBER),
                wvc.Property(name="load_fraction", data_type=wvc.DataType.NUMBER),
                # Output results
                wvc.Property(name="result_json", data_type=wvc.DataType.TEXT),
            ],
            vectorizer_config=wvc.Configure.Vectorizer.none(),
        )

    # ==========================================================================
    # Tool Tracking (External Dependencies)
    # ==========================================================================

    # 20. Tool (tracks external tools and libraries used by modules)
    if not client.collections.exists("Tool"):
        client.collections.create(
            name="Tool",
            description="An external tool or library used by orchestrator modules",
            properties=[
                wvc.Property(name="tool_id", data_type=wvc.DataType.TEXT, skip_vectorization=True),
                wvc.Property(name="name", data_type=wvc.DataType.TEXT),  # Display name
                wvc.Property(name="version", data_type=wvc.DataType.TEXT),
                wvc.Property(
                    name="category", data_type=wvc.DataType.TEXT
                ),  # optimization, cfd, fea, ml, utility
                wvc.Property(
                    name="import_pattern", data_type=wvc.DataType.TEXT
                ),  # regex for detection
                wvc.Property(name="documentation_url", data_type=wvc.DataType.TEXT),
                wvc.Property(name="license", data_type=wvc.DataType.TEXT),
            ],
            vectorizer_config=wvc.Configure.Vectorizer.none(),
        )

    # 20b. Add uses_tools reference from Module to Tool
    modules = client.collections.get("Module")
    current_refs = [r.name for r in modules.config.get().references]
    if "uses_tools" not in current_refs:
        modules.config.add_reference(
            wvc.ReferenceProperty(name="uses_tools", target_collection="Tool")
        )
