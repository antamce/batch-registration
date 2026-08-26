# Paired TIFF stack registration with ImageJ StackReg

`register_tiff_stacks.py` recursively finds paired 2D `ChanA*.tif[f]` and
`ChanB*.tif[f]` files and naturally sorts them. For each specimen it builds two
true ImageJ stacks, registers ChanB sequentially with translation-only
StackReg/TurboReg, and uses MultiStackReg to apply the identical transforms to
ChanA.

The final files are uncompressed, single-channel OME-BigTIFF Z-stacks. The
script checks that each output has one `ZYX` series, the expected page count and
data type, BigTIFF format, OME metadata, and no compression before marking the
specimen complete.

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
