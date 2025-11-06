"""Utility to load dataset CSV files into pandas DataFrames.

The script searches the `dataset/` directory for CSV files (recursively)
and loads each CSV into a pandas.DataFrame. Use the `load_all_csvs`
function to get a mapping of relative CSV paths -> DataFrames.

Run as a script to print a short summary (file, shape, columns).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Callable, Iterable, List, Optional, Tuple

try:
	import pandas as pd
except Exception as exc:  # pragma: no cover - runtime dependency
	raise ImportError(
		"pandas is required to run this script. Install it with `pip install pandas`."
	) from exc

try:
    import numpy as np
except Exception as exc:  # pragma: no cover - runtime dependency
    raise ImportError("numpy is required to load images as arrays. Install with `pip install numpy`") from exc


def load_all_csvs(base_dir: str | Path = "dataset") -> Dict[str, pd.DataFrame]:
	"""Find and load all CSV files under `base_dir` into pandas DataFrames.

	Args:
		base_dir: Path to the dataset folder (default: 'dataset').

	Returns:
		A dict mapping the CSV file's relative path (relative to base_dir)
		to a pandas.DataFrame loaded from that CSV.

	Behavior / notes:
	- If no CSV files are found, an empty dict is returned.
	- If a CSV fails to parse, a warning is printed and that file is skipped.
	"""
	base = Path(base_dir)
	results: Dict[str, pd.DataFrame] = {}

	if not base.exists():
		print(f"Warning: base dataset directory '{base}' does not exist.")
		return results

	csv_files = sorted(base.rglob("*.csv"))
	if not csv_files:
		print(f"No CSV files found under '{base}'.")
		return results

	for p in csv_files:
		rel = str(p.relative_to(base))
		try:
			df = pd.read_csv(p)
			results[rel] = df
		except Exception as e:
			print(f"Failed to read '{p}': {e}")

	return results


def summary_str(dfs: Dict[str, pd.DataFrame]) -> str:
	"""Return a compact multi-line summary for a mapping of DataFrames."""
	if not dfs:
		return "(no DataFrames loaded)"

	lines = []
	for rel, df in dfs.items():
		# show up to first 5 columns for brevity
		cols = list(df.columns)
		col_preview = ", ".join(cols[:5]) + (", ..." if len(cols) > 5 else "")
		lines.append(f"{rel}: rows={len(df):,}, cols={len(cols)} -> {col_preview}")
	return "\n".join(lines)


if __name__ == "__main__":
	# Quick runnable check when executed directly
	dataset_dir = Path(__file__).parent / "dataset"

	def load_split(split_dir: str | Path, load_images: bool = False) -> pd.DataFrame:
		"""Load a single split folder containing an `_classes.csv` and images.

		Returns a DataFrame with columns: 'filename', <class columns...>, 'image_path',
		and optionally 'image' (PIL.Image) when load_images=True.
		"""
		split = Path(split_dir)
		csv_path = split / "_classes.csv"
		if not csv_path.exists():
			raise FileNotFoundError(f"Expected _classes.csv in {split} but not found")

		df = pd.read_csv(csv_path)
		if "filename" not in df.columns:
			raise ValueError(f"_classes.csv at {csv_path} has no 'filename' column")

		# create full image path column
		df["image_path"] = df["filename"].apply(lambda fn: str((split / fn).resolve()))

		# check existence
		missing = [p for p in df["image_path"] if not Path(p).exists()]
		if missing:
			print(f"Warning: {len(missing)} images listed in {csv_path} were not found on disk")

		if load_images:
			try:
				from PIL import Image
			except Exception as exc:  # pragma: no cover - optional dependency
				raise ImportError("Pillow is required to load images. Install with `pip install pillow`") from exc

			images = []
			for p in df["image_path"]:
				try:
					images.append(Image.open(p).convert("RGB"))
				except Exception:
					images.append(None)
			df["image"] = images

		return df


	def load_all_splits(base_dir: str | Path = dataset_dir, splits=("train", "valid", "test"), load_images: bool = False) -> Dict[str, pd.DataFrame]:
		"""Load `_classes.csv` and image paths for each split under base_dir.

		Returns a dict mapping split name -> DataFrame.
		"""
		base = Path(base_dir)
		out: Dict[str, pd.DataFrame] = {}
		for s in splits:
			split_dir = base / s
			if not split_dir.exists():
				print(f"Skipping missing split folder: {split_dir}")
				continue
			try:
				out[s] = load_split(split_dir, load_images=load_images)
			except Exception as e:
				print(f"Failed loading split {s}: {e}")
		return out


	# load (only paths by default to avoid heavy memory use)
	splits = load_all_splits(dataset_dir, load_images=False)

	print("Loaded dataset splits summary:\n")
	for name, df in splits.items():
		classes = [c for c in df.columns if c not in ("filename", "image_path", "image")]
		print(f"{name}: rows={len(df):,}, class_cols={len(classes)}, image_path_exists={df['image_path'].apply(lambda p: Path(p).exists()).sum()}")


	# --- image loading helpers (MNIST-style eager loader + lazy wrapper) ---

	def get_class_columns(df: pd.DataFrame) -> List[str]:
		return [c for c in df.columns if c not in ("filename", "image_path", "image")]


	def images_to_numpy(
		df: pd.DataFrame,
		image_col: str = "image_path",
		size: Optional[Tuple[int, int]] = None,
		as_gray: bool = False,
		normalize: bool = True,
		dtype: type = np.float32,
		show_progress: bool = False,
	) -> Tuple[np.ndarray, np.ndarray, List[str]]:
		"""Eagerly load images listed in `df[image_col]` into a NumPy array (MNIST-style).

		Returns (X, y, class_names) where:
		  - X: np.ndarray, shape (N, H, W, C) with C=1 if as_gray else 3
		  - y: np.ndarray, shape (N, num_classes) multi-hot matrix (ints)
		  - class_names: list of class column names in the same order as y columns

		Notes:
		  - If `size` is provided, images are resized to that (width, height) using Pillow.
		  - `normalize` scales pixel values to [0,1] floats when True.
		"""
		try:
			from PIL import Image
		except Exception as exc:  # pragma: no cover - optional dependency
			raise ImportError("Pillow is required to load images. Install with `pip install pillow`") from exc

		paths = list(df[image_col].astype(str))
		class_cols = get_class_columns(df)
		y_arr = df[class_cols].to_numpy(dtype=int)

		imgs: List[np.ndarray] = []
		total = len(paths)
		for i, p in enumerate(paths):
			if show_progress and i % 100 == 0:
				print(f"loading image {i}/{total}")
			try:
				img = Image.open(p)
				img = img.convert("L") if as_gray else img.convert("RGB")
				if size is not None:
					img = img.resize(size, Image.BILINEAR)
				arr = np.asarray(img, dtype=dtype)
				if as_gray:
					# shape (H,W) -> (H,W,1)
					arr = arr.reshape(( *arr.shape, 1))
				imgs.append(arr)
			except Exception:
				# on failure append a zero array of the right shape (if possible)
				if size is None:
					raise
				c = 1 if as_gray else 3
				imgs.append(np.zeros((size[1], size[0], c), dtype=dtype))

		X = np.stack(imgs, axis=0)
		if normalize:
			X = X / 255.0

		return X, y_arr, class_cols


	class LazyImageDataset:
		"""Simple lazy dataset that reads images on-the-fly from a DataFrame.

		Behaves like a small MNIST-style dataset: __len__ and __getitem__.
		__getitem__ returns (image_array, label_array).
		"""

		def __init__(
			self,
			df: pd.DataFrame,
			image_col: str = "image_path",
			size: Optional[Tuple[int, int]] = None,
			as_gray: bool = False,
			normalize: bool = True,
			transform: Optional[Callable] = None,
		) -> None:
			self.df = df.reset_index(drop=True)
			self.image_col = image_col
			self.size = size
			self.as_gray = as_gray
			self.normalize = normalize
			self.transform = transform
			self.class_cols = get_class_columns(df)

		def __len__(self) -> int:
			return len(self.df)

		def __getitem__(self, idx: int):
			try:
				from PIL import Image
			except Exception:
				raise ImportError("Pillow is required to load images. Install with `pip install pillow`")

			row = self.df.iloc[idx]
			p = str(row[self.image_col])
			img = Image.open(p)
			img = img.convert("L") if self.as_gray else img.convert("RGB")
			if self.size is not None:
				img = img.resize(self.size, Image.BILINEAR)
			arr = np.asarray(img, dtype=np.float32)
			if self.as_gray and arr.ndim == 2:
				arr = arr.reshape(( *arr.shape, 1))
			if self.normalize:
				arr = arr / 255.0

			if self.transform is not None:
				arr = self.transform(arr)

			labels = row[self.class_cols].to_numpy(dtype=int)
			return arr, labels

	# brief hint for the user how to get MNIST-like arrays
	print("\nHints: call `images_to_numpy(df, size=(28,28))` to get (X,y) arrays similar to MNIST, or create `LazyImageDataset(df)` for on-the-fly loading.")


