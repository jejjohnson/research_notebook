---
title: "MARS Satellite Constellation — Technical Characteristics"
short_title: Satellite catalog
subject: plumax — observation side
authors:
  - name: J. Emmanuel Johnson
    affiliations: [UNEP, IMEO, MARS]
    orcid: 0000-0002-6739-0053
    email: jemanjohnson34@gmail.com
license: CC-BY-4.0
keywords: [methane, satellite, MARS, TROPOMI, GHGSat, EMIT, Carbon Mapper, MethaneSAT, point-source imager, area-flux mapper, XCH4, SWIR]
numbering:
  enumerator: "S%s"
---

# MARS Satellite Constellation — Technical Characteristics

The **Methane Alert and Response System (MARS)** integrates data from over 35 satellites, split into two functional categories.

:::{note} Detection categories
:class: dropdown

- **Point Source Imagers (PSI)** — high spatial resolution (20–60 m), tasked or systematic imaging, sensitivity to individual facility plumes (~100–1{,}000 kg/h detection floor).
- **Area Flux Mappers (AFM)** — wide-swath column retrievals of XCH₄ (in ppb), enabling regional and national budget attribution.
:::

The table below is interactive: filter by category, era, or free-text search across satellite name, instrument, and spectral method. The HTML, CSS, and filter logic are loaded from a sibling file ([`satellites_table.html`](satellites_table.html)) so the rendering is identical to the legacy interactive cell — see the mystmd guide on [including tables from file](https://mystmd.org/guide/tables#include-tables-from-file) for the directive pattern.

::::{table} MARS satellite constellation — past, present, and future missions tasked for methane retrieval, with spatial / spectral / SNR characteristics and per-row references into `methane.bib`.
:label: tbl-mars-satellites

:::{include} satellites_table.html
:::

::::

## Notes

:::{important} Detection philosophy
The two categories reflect fundamentally different retrieval strategies:

- PSI instruments use high spatial resolution to spatially resolve individual facility plumes, with SNR and spectral resolution tuned to the 1.65 µm SWIR absorption feature.
- AFM instruments retrieve column-averaged dry-air mole fractions (XCH₄ in ppb) across wide swaths, enabling regional budget inversion.
:::

:::{note} Spectral windows
Nearly all instruments use either the **1.60–1.68 µm** window (lower absorption depth, cleaner for point-source retrieval) or the **2.3 µm** band (deeper absorption, preferred for regional XCH₄ columns, used by TROPOMI/Sentinel-5; see {cite:p}`s5p_tropomi`). MERLIN ({cite:p}`merlin`) is unique in using an IPDA lidar with matched on/off wavelengths at 1645 nm rather than passive solar backscatter.
:::

:::{caution} Caveats
- SNR figures are scene- and band-dependent; many instruments publish full spectrally-resolved SNR curves.
- Revisit times for tasked instruments (GHGSat {cite:p}`ghgsat`, PRISMA {cite:p}`prisma`, Carbon Mapper {cite:p}`carbon_mapper`) depend entirely on operator scheduling.
- Future-mission specs (SBG {cite:p}`sbg`, CHIME {cite:p}`chime`, CO2M {cite:p}`co2m`, MERLIN {cite:p}`merlin`, TANGO {cite:p}`tango`) are from pre-launch design documents and subject to revision.
- GOSAT-GW ({cite:p}`gosat_gw`) launched December 2023 but was still in commissioning as of early 2024; some specs are pre-launch.
- EMIT ({cite:p}`emit`) revisit is irregular due to ISS precession drift — coverage varies from 1–4 passes/month depending on latitude.
:::

## References

Per-row citations are indexed in the table's `Ref` column. Full bibliographic entries live in `methane.bib`; the keys used here are: {cite:p}`landsat45,landsat7,sentinel2,ghgsat,landsat89,carbon_mapper,gaofen5,ziyuan1,enmap,prisma,emit,viirs,sentinel3,goes_abi,mtg,tango,co2image,sbg,chime,sciamachy,methanesat,gosat,s5p_tropomi,gosat_gw,s5_metop_sg,co2m,merlin`.
