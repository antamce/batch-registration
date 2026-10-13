# Paired TIFF stack registration with ImageJ StackReg

`register_tiff_stacks.py` registers paired two-channel TIFF data. By default it
recursively finds paired 2D `ChanA*.tif[f]` and `ChanB*.tif[f]` files and builds
them into stacks. With `--prestacked`, it instead finds existing `g`/`r` TIFF
stacks. With `--sepfolder`, it pairs individual green/red planes acquired in
separate folders. Registration uses translation-only StackReg/TurboReg, and
MultiStackReg applies the identical transforms to the other channel.

The default final files are uncompressed, single-channel OME-BigTIFF Z-stacks.
The script checks each output's format, `ZYX` shape, page count, data type, and
compression before marking the specimen complete.

## Requirements

Python 3.11 or newer is recommended:

```powershell
python -m pip install -r requirements.txt
```

The default ImageJ installation is the `ImageJ` directory beside the script.
It must contain its bundled Java and these plugins:

```text
ImageJ/plugins/StackReg_.jar
ImageJ/plugins/TurboReg_.jar
ImageJ/plugins/MultiStackRegistration_.jar
```

MultiStackReg is the companion that saves ChanB's StackReg/TurboReg
transformations and applies those same transformations to ChanA.

## Run

Pass an experiment root (or one specimen folder) and a separate output folder:

```powershell
python register_tiff_stacks.py "D:\data\exp" "E:\registered"
```

Two specimen folders are processed simultaneously by default. Each ImageJ
worker may use up to 8 GB of Java heap. Both settings are configurable:

```powershell
python register_tiff_stacks.py "D:\data\exp" "E:\registered" --workers 2 --imagej-memory 8g
```

Use another ImageJ installation if needed:

```powershell
python register_tiff_stacks.py "D:\data\exp" "E:\registered" --imagej-dir "C:\Fiji.app"
```

### Existing g/r stacks

Use `--prestacked` when the input already contains stacks. A pair must be in
the same folder and have identical names except for the final `g` or `r` before
`.tif`/`.tiff`, for example `2306_1abg.tif` and `2306_1abr.tif`:

```powershell
python register_tiff_stacks.py "D:\data\stacks" "E:\registered" --prestacked
```

The input root itself and all subfolders are searched. Single-page TIFFs are
ignored. Each pair must contain one grayscale `ZYX` series with matching page
count, dimensions, and data type. Invalid pairs are reported and skipped while
other valid pairs continue. Relative input directories are preserved under the
output directory to prevent same-name collisions.

By default, the `r` stack is aligned and the same transformations are applied
to `g`. Invert that choice with:

```powershell
python register_tiff_stacks.py "D:\data\stacks" "E:\registered" --prestacked --align-g
```

Prestacked output names use the input basename, such as:

```text
2306_1abg_registered.tif
2306_1abr_registered.tif
shifts/
  2306_1ab_shifts.csv
```

The shift CSV identifies stack slices as `filename.tif#page=N`, with page
numbers starting at 1. Embedded OME or ImageJ calibration is used first;
`Experiment.xml` is only a fallback.

### Output TIFF format

Outputs are uncompressed OME-BigTIFF by default. To write uncompressed classic
ImageJ TIFF stacks instead, use `--imagej-tiff` in either input mode:

```powershell
python register_tiff_stacks.py "D:\data\stacks" "E:\registered" --prestacked --imagej-tiff
```

Classic TIFF has a 4 GB size limit, so this option should only be used when
each resulting file will remain below that limit.

### Channels acquired in separate folders

Use `--sepfolder` when green and red planes are stored in sibling acquisition
folders:

```text
<input>/cafsynpo03062/kab7g2/ChanA_0001_0001_0001_0001.tif
<input>/cafsynpo03062/kab7r2/ChanB_0001_0001_0001_0001.tif
```

```powershell
python register_tiff_stacks.py "D:\data\separate" "E:\registered" --sepfolder
```

Folder names must have equal length and differ at exactly one character: `g`
for the folder containing `ChanA` planes and `r` for the folder containing
`ChanB` planes. The marker may occur anywhere, so `kab7g2`/`kab7r2` is valid.
After the leading `ChanA`/`ChanB` is removed, paired TIFF filenames must match
exactly, including their extension.

All folder names below the supplied input root are flattened into the output
names. The example above produces:

```text
cafsynpo03062_kab7g2_registered.tif
cafsynpo03062_kab7r2_registered.tif
shifts/
  cafsynpo03062_kab72_shifts.csv
```

By default, `r` is aligned and its transformations are applied to `g`. Add
`--align-g` to invert that behavior. `--imagej-tiff` is also supported.
`--sepfolder` and `--prestacked` are mutually exclusive.

Each channel's own XY and Z calibration is preserved in its corresponding
output. Differences are logged as warnings but do not stop registration.
Folders without an identifiable channel marker are ignored. An identifiable
`g` or `r` folder without its partner is warned about and skipped; the other
valid pairs still run, and the process exits nonzero after the batch finishes.

Check recursive discovery, natural sorting, and channel pairing without reading
pixels or running ImageJ:

```powershell
python register_tiff_stacks.py "D:\data\exp" "E:\registered" --dry-run
```

Completed specimens are skipped when both registered TIFFs and their shift CSV
exist. Use `--overwrite` to reprocess them. A failed specimen is logged and the
batch continues with the next one. Partial final files and per-specimen ImageJ
working files are removed after success or failure.

## Folder layout and names

Input is expected to follow:

```text
<experiment day>/<experimental group>/<specimen>/<Chan*.tif>
```

For `1606/glu0/Untitled002`, the output directory contains:

```text
1606_glu0_Untitled002_ChanA_registered.tif
1606_glu0_Untitled002_ChanB_registered.tif
registration.log
shifts/
  1606_glu0_Untitled002_shifts.csv
```

The CSV preserves the source filenames and records the consecutive and
accumulated X/Y translations reported by MultiStackReg. Positive X is right and
positive Y is down in the output image coordinate system.

Usable TIFF X/Y resolution tags are copied. Z spacing is read from
`Experiment.xml` (`ZStage/@stepSizeUM`); missing or invalid Z metadata is logged
but does not fail the specimen.

## Registration details

ImageJ performs StackReg's sequential translation alignment with slice 1 as the
anchor. StackReg/TurboReg can estimate subpixel translations and resamples the
images when applying them; exposed borders are black (zero for these images).
There are no Python shift-size or confidence guards.

The standard StackReg command cannot export and replay transforms on another
stack. MultiStackReg uses the same StackReg/TurboReg registration machinery and
adds the required “align ChanB, apply to ChanA” workflow.

## Fiji validation

Open a result with **Plugins → Bio-Formats → Bio-Formats Importer**. It should
report one series, one channel, `Z/Y/X` axes, the original data type and XY
dimensions, and one Z-slice per source pair. BigTIFF avoids the classic TIFF
4 GB limit.

Because processing is uncompressed, temporary disk usage is substantial while
a specimen is active: source-derived ImageJ stacks, plugin outputs, and final
partial outputs coexist. The hidden `.stackreg_work` directory under the output
folder is cleaned per specimen.
