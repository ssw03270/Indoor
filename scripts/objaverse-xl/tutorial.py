import objaverse
print(objaverse.__version__)

import objaverse.xl as oxl

annotations = oxl.get_annotations(
    download_dir="~/.objaverse" # default download directory
)
print(annotations)
print(annotations["source"].value_counts())
print(annotations["fileType"].value_counts())

alignment_annotations = oxl.get_alignment_annotations(
    download_dir="~/.objaverse" # default download directory
)
print(alignment_annotations)