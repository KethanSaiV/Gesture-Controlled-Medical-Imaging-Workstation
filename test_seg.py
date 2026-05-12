import tempfile, os, SimpleITK as sitk, subprocess, sys

# Find DICOM files recursively from project root
project = r'C:\Users\jsvis\OneDrive\Desktop\final\Gesture-Controlled-Medical-Imaging-Workstation'

print('Searching for DICOM files...')
dicom_dir = None
for root, dirs, files in os.walk(project):
    dcm_files = [f for f in files if f.endswith('.dcm') or (len(f) < 20 and '.' not in f)]
    if len(dcm_files) > 5:
        dicom_dir = root
        print(f'Found {len(dcm_files)} files in: {root}')
        break

if dicom_dir is None:
    print('ERROR: No DICOM folder found. Contents of project:')
    for root, dirs, files in os.walk(project):
        print(f'  {root}: {len(files)} files')
    sys.exit(1)

tmp = tempfile.mkdtemp(prefix='totalseg_test_')
nii_in = os.path.join(tmp, 'input.nii.gz')
out_dir = os.path.join(tmp, 'segs')
os.makedirs(out_dir, exist_ok=True)

reader = sitk.ImageSeriesReader()
series_ids = reader.GetGDCMSeriesIDs(dicom_dir)
if not series_ids:
    print('ERROR: SimpleITK found no DICOM series in:', dicom_dir)
    sys.exit(1)

files = reader.GetGDCMSeriesFileNames(dicom_dir, series_ids[0])
reader.SetFileNames(files)
sitk.WriteImage(reader.Execute(), nii_in)
print('NIfTI written:', nii_in)
print('Running TotalSegmentator...')
print('='*60)

result = subprocess.run(
    [sys.executable, '-m', 'totalsegmentator.bin.TotalSegmentator',
     '-i', nii_in, '-o', out_dir, '-ta', 'total_mr', '--ml', '-d', 'cpu', '--fast'],
)

print('='*60)
print('Return code:', result.returncode)
print('Files in output dir:')
for f in os.listdir(out_dir):
    print(' ', f)
if not os.listdir(out_dir):
    print('  (empty)')