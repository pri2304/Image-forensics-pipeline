import io

from pyexiv2 import ImageMetadata
from PIL import Image, ImageCms

image_path = "facebook.jpg"
with open(image_path, 'rb') as f:
    soi = f.read(2)
    if soi == b'\xFF\xD8':
        print("True")
    else:
        print("False")

    f.seek(-2, 2)
    eoi = f.read(2)
    if eoi == b'\xFF\xD9':
        print("True")
    else:
        print("False")

with Image.open(image_path) as img:
    quantization_tables = getattr(img, 'quantization', None)
    print(quantization_tables)
    if hasattr(img, 'applist'):
        print(len(img.applist))
        for marker_name, data in img.applist:
            if marker_name == 'APP0':
                print(f"Found APP0 (Size = {len(data)} bytes)")

            if marker_name == 'APP1':
                print(f"Found APP1 (Size = {len(data)} bytes)")

            if marker_name == 'APP2':
                print(f"Found APP2 (Size = {len(data)} bytes)")

            if marker_name == 'APP3':
                print(f"Found APP2 (Size = {len(data)} bytes)")

            if marker_name == 'APP4':
                print(f"Found APP4 (Size = {len(data)} bytes)")

            if marker_name == 'APP5':
                print(f"Found APP5 (Size = {len(data)} bytes)")

            if marker_name == 'APP6':
                print(f"Found APP6 (Size = {len(data)} bytes)")

            if marker_name == 'APP7':
                print(f"Found APP7 (Size = {len(data)} bytes)")

            if marker_name == 'APP8':
                print(f"Found APP8 (Size = {len(data)} bytes)")

            if marker_name == 'APP9':
                print(f"Found APP9 (Size = {len(data)} bytes)")

            if marker_name == 'APP10':
                print(f"Found APP10 (Size = {len(data)} bytes)")

            if marker_name == 'APP11':
                print(f"Found APP11 (Size = {len(data)} bytes)")

            if marker_name == 'APP12':
                print(f"Found APP12 (Size = {len(data)} bytes)")

            if marker_name == 'APP13':
                print(f"Found APP13 (Size = {len(data)} bytes)")

            if marker_name == 'APP14':
                print(f"Found APP14 (Size = {len(data)} bytes)")

            if marker_name == 'APP15':
                print(f"Found APP15 (Size = {len(data)} bytes)")

    raw_icc_profile = img.info.get('icc_profile')
    if raw_icc_profile:
        profile = ImageCms.getOpenProfile(io.BytesIO(raw_icc_profile))
        profile_name = profile.profile.profile_description
        print(profile_name)
    else:
        print('no icc profile')


metadata = ImageMetadata(image_path)
metadata.read()
exif_keys = metadata.exif_keys
xmp_keys = metadata.xmp_keys
iptc_keys = metadata.iptc_keys
'''print(xmp_keys)
print(exif_keys)
print(iptc_keys)'''
for key in exif_keys:
    try:
        print(f'for tag {metadata[key]} value is {metadata[key].raw_value}')
    except KeyError as e:
        print(f'could not read value of {key}')
        print(f'error {e}\n')
        continue
for key in xmp_keys:
    try:
        print(f'for tag {metadata[key]} value is {metadata[key].raw_value}')
    except KeyError as e:
        print(f'could not read value of {key}')
        print(f'error {e}\n')
        continue
for key in iptc_keys:
    try:
        print(f'for tag {metadata[key]} value is {metadata[key].raw_value}')
    except KeyError as e:
        print(f'could not read value of {key}')
        print(f'error {e}\n')
        continue