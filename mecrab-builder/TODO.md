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
- [ ] Wikipedia abstract integration
- [ ] DBpedia support
- [ ] Custom ontology import

### Output Formats
- [ ] Direct sys.dic generation
- [ ] CSV with embedded URIs

### Performance
- [ ] Parallel processing with rayon
- [ ] Incremental index updates
- [ ] Delta processing for updates
- [ ] Memory-efficient streaming

### Quality
- [ ] Confidence calibration
- [ ] Entity type filtering
- [ ] POS-based URI filtering
- [ ] Duplicate detection
