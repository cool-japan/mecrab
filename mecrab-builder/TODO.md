# mecrab-builder TODO

## Completed
- [x] Basic project structure
- [x] Wikidata JSON streaming parser
- [x] WikidataIndex (surface → URIs)
- [x] Dictionary CSV merging
- [x] Gzip streaming decompression
- [x] CLI commands (download, index, enrich)
- [x] SemanticPool binary output (MCV1 format)
- [x] Comprehensive test suite (5 tests)
- [x] Binary serialization roundtrip tests

## Planned

### Data Sources
- [x] Wikipedia abstract integration
- [x] DBpedia support
- [x] Online entity resolution (Wikidata API fallback)
- [x] Custom ontology import (CSV, JSON, RDF/OWL)

### Output Formats
- [x] Direct sys.dic generation
- [x] CSV with embedded URIs

### Performance
- [x] Parallel processing with rayon (chunked parallel JSON parse in build_wikidata_index)
- [x] Incremental index updates
- [x] Delta processing for updates
- [x] Memory-efficient streaming

### Quality
- [x] Confidence calibration (calibrated_confidence + recalibrate() ambiguity penalty)
- [x] Entity type filtering (P31 claims, entity_type_filter in BuildConfig)
- [x] POS-based URI filtering (pos_to_allowed_entity_types in merge_dictionary)
- [x] Duplicate detection (max-confidence HashMap dedup in WikidataIndex)
