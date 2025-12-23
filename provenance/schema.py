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
