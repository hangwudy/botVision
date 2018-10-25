import tensorflow as tf

if not faces_only:
if mask_type == 'numerical':
    mask_stack = np.stack(masks).astype(np.float32)
    masks_flattened = np.reshape(mask_stack, [-1])
    feature_dict['image/object/mask'] = (
        dataset_util.float_list_feature(masks_flattened.tolist()))
elif mask_type == 'png':
    encoded_mask_png_list = []
    for mask in masks:
    img = PIL.Image.fromarray(mask)
    output = io.BytesIO()
    img.save(output, format='PNG')
    encoded_mask_png_list.append(output.getvalue())
    feature_dict['image/object/mask'] = (
        dataset_util.bytes_list_feature(encoded_mask_png_list))