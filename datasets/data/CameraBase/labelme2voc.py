import argparse, os, glob, json, base64, io, math
from PIL import Image, ImageDraw

COLOR_MAP = [
    0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128, 64, 0, 0, 192, 0, 0,
    64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128, 64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128,
    192, 0, 0, 64, 128, 128, 64, 128, 0, 192, 128, 128, 192, 128, 64, 64, 0, 192, 64, 0, 64, 192, 0, 192, 192, 0, 64,
    64, 128, 192, 64, 128, 64, 192, 128, 192, 192, 128, 0, 0, 64, 128, 0, 64, 0, 128, 64, 128, 128, 64, 0, 0, 192, 128,
    0, 192, 0, 128, 192, 128, 128, 192, 64, 0, 64, 192, 0, 64, 64, 128, 64, 192, 128, 64, 64, 0, 192, 192, 0, 192, 64,
    128, 192, 192, 128, 192, 0, 64, 64, 128, 64, 64, 0, 192, 64, 128, 192, 64, 0, 64, 192, 128, 64, 192, 0, 192, 192,
    128, 192, 192, 64, 64, 64, 192, 64, 64, 64, 192, 64, 192, 192, 64, 64, 64, 192, 192, 64, 192, 64, 192, 192, 192,
    192, 192, 32, 0, 0, 160, 0, 0, 32, 128, 0, 160, 128, 0, 32, 0, 128, 160, 0, 128, 32, 128, 128, 160, 128, 128, 96, 0,
    0, 224, 0, 0, 96, 128, 0, 224, 128, 0, 96, 0, 128, 224, 0, 128, 96, 128, 128, 224, 128, 128, 32, 64, 0, 160, 64, 0,
    32, 192, 0, 160, 192, 0, 32, 64, 128, 160, 64, 128, 32, 192, 128, 160, 192, 128, 96, 64, 0, 224, 64, 0, 96, 192, 0,
    224, 192, 0, 96, 64, 128, 224, 64, 128, 96, 192, 128, 224, 192, 128, 32, 0, 64, 160, 0, 64, 32, 128, 64, 160, 128,
    64, 32, 0, 192, 160, 0, 192, 32, 128, 192, 160, 128, 192, 96, 0, 64, 224, 0, 64, 96, 128, 64, 224, 128, 64, 96, 0,
    192, 224, 0, 192, 96, 128, 192, 224, 128, 192, 32, 64, 64, 160, 64, 64, 32, 192, 64, 160, 192, 64, 32, 64, 192, 160,
    64, 192, 32, 192, 192, 160, 192, 192, 96, 64, 64, 224, 64, 64, 96, 192, 64, 224, 192, 64, 96, 64, 192, 224, 64, 192,
    96, 192, 192, 224, 192, 192, 0, 32, 0, 128, 32, 0, 0, 160, 0, 128, 160, 0, 0, 32, 128, 128, 32, 128, 0, 160, 128,
    128, 160, 128, 64, 32, 0, 192, 32, 0, 64, 160, 0, 192, 160, 0, 64, 32, 128, 192, 32, 128, 64, 160, 128, 192, 160,
    128, 0, 96, 0, 128, 96, 0, 0, 224, 0, 128, 224, 0, 0, 96, 128, 128, 96, 128, 0, 224, 128, 128, 224, 128, 64, 96, 0,
    192, 96, 0, 64, 224, 0, 192, 224, 0, 64, 96, 128, 192, 96, 128, 64, 224, 128, 192, 224, 128, 0, 32, 64, 128, 32, 64,
    0, 160, 64, 128, 160, 64, 0, 32, 192, 128, 32, 192, 0, 160, 192, 128, 160, 192, 64, 32, 64, 192, 32, 64, 64, 160,
    64, 192, 160, 64, 64, 32, 192, 192, 32, 192, 64, 160, 192, 192, 160, 192, 0, 96, 64, 128, 96, 64, 0, 224, 64, 128,
    224, 64, 0, 96, 192, 128, 96, 192, 0, 224, 192, 128, 224, 192, 64, 96, 64, 192, 96, 64, 64, 224, 64, 192, 224, 64,
    64, 96, 192, 192, 96, 192, 64, 224, 192, 192, 224, 192, 32, 32, 0, 160, 32, 0, 32, 160, 0, 160, 160, 0, 32, 32, 128,
    160, 32, 128, 32, 160, 128, 160, 160, 128, 96, 32, 0, 224, 32, 0, 96, 160, 0, 224, 160, 0, 96, 32, 128, 224, 32,
    128, 96, 160, 128, 224, 160, 128, 32, 96, 0, 160, 96, 0, 32, 224, 0, 160, 224, 0, 32, 96, 128, 160, 96, 128, 32,
    224, 128, 160, 224, 128, 96, 96, 0, 224, 96, 0, 96, 224, 0, 224, 224, 0, 96, 96, 128, 224, 96, 128, 96, 224, 128,
    224, 224, 128, 32, 32, 64, 160, 32, 64, 32, 160, 64, 160, 160, 64, 32, 32, 192, 160, 32, 192, 32, 160, 192, 160,
    160, 192, 96, 32, 64, 224, 32, 64, 96, 160, 64, 224, 160, 64, 96, 32, 192, 224, 32, 192, 96, 160, 192, 224, 160,
    192, 32, 96, 64, 160, 96, 64, 32, 224, 64, 160, 224, 64, 32, 96, 192, 160, 96, 192, 32, 224, 192, 160, 224, 192, 96,
    96, 64, 224, 96, 64, 96, 224, 64, 224, 224, 64, 96, 96, 192, 224, 96, 192, 96, 224, 192, 224, 224, 192
]

IMAGE_SETS_DIR_NAME = 'ImageSets'
SEGMENTATION_DIR_NAME = 'Segmentation'
JPEG_IMAGES_DIR_NAME = "JPEGImages"
SEGMENTATION_CLASS_DIR_NAME = 'SegmentationClass'
SEGMENTATION_CLASS_RAW_DIR_NAME = 'SegmentationClassRaw'
SEGMENTATION_CLASS_VISUALIZATION_DIR_NAME = 'SegmentationClassVisualization'
LABELS_TEXT_FILE_NAME = 'labels.txt'
TRAINVAL_TXT_FILE_NAME = "trainval.txt"
TRAIN_TXT_FILE_NAME = "train.txt"
VAL_TXT_FILE_NAME = "val.txt"
BORDER_WIDTH = 4
BORDER_COLOR = 255


class ShapeTypes:
    CIRCLE = "circle"
    POLYGON = "polygon"
    RECTANGLE = 'rectangle'


def check_to_mkdir(p):
    if not os.path.exists(p):
        os.mkdir(p)


def get_image_sets_dir(base_output_dir):
    return os.path.join(base_output_dir, IMAGE_SETS_DIR_NAME)


def get_segmentation_dir(base_output_dir):
    return os.path.join(base_output_dir, IMAGE_SETS_DIR_NAME, SEGMENTATION_DIR_NAME)


def get_jpeg_images_dir(base_output_dir):
    return os.path.join(base_output_dir, JPEG_IMAGES_DIR_NAME)


def get_segmentation_class_dir(base_output_dir):
    return os.path.join(base_output_dir, SEGMENTATION_CLASS_DIR_NAME)


def get_segmentation_class_raw_dir(base_output_dir):
    return os.path.join(base_output_dir, SEGMENTATION_CLASS_RAW_DIR_NAME)


def get_segmentation_class_visualization_dir(base_output_dir):
    return os.path.join(base_output_dir, SEGMENTATION_CLASS_VISUALIZATION_DIR_NAME)


def get_labels_text_file(base_output_dir):
    return os.path.join(base_output_dir, LABELS_TEXT_FILE_NAME)


def get_trainval_text_file(base_output_dir):
    return os.path.join(get_segmentation_dir(base_output_dir), TRAINVAL_TXT_FILE_NAME)


def get_train_text_file(base_output_dir):
    return os.path.join(get_segmentation_dir(base_output_dir), TRAIN_TXT_FILE_NAME)


def get_val_text_file(base_output_dir):
    return os.path.join(get_segmentation_dir(base_output_dir), VAL_TXT_FILE_NAME)


def check_to_make_output_dirs(output_dir):
    check_to_mkdir(output_dir)
    check_to_mkdir(get_image_sets_dir(output_dir))
    check_to_mkdir(get_segmentation_dir(output_dir))
    check_to_mkdir(get_jpeg_images_dir(output_dir))
    check_to_mkdir(get_segmentation_class_dir(output_dir))
    check_to_mkdir(get_segmentation_class_raw_dir(output_dir))
    check_to_mkdir(get_segmentation_class_visualization_dir(output_dir))


def translate_points(points):
    new_arr = []
    for p in points:
        new_arr.append((p[0], p[1]))
    return new_arr


labels = ["_background_"]


def get_label_color(label):
    if label in labels:
        color_index = labels.index(label)
    else:
        labels.append(label)
        color_index = len(labels) - 1
    return color_index


def get_rgb_by_p_index(color_index):
    start = color_index * 3
    return COLOR_MAP[start], COLOR_MAP[start + 1], COLOR_MAP[start + 2]


def draw_polygon(seg_class_img, shape, seg_class_visualization_img=None, seg_class_raw_img=None):
    seg_class_img_draw = ImageDraw.Draw(seg_class_img)
    points = translate_points(shape['points'])
    fill_color = get_label_color(shape['label'])

    seg_class_raw_img_draw = None
    if seg_class_raw_img:
        seg_class_raw_img_draw = ImageDraw.Draw(seg_class_raw_img)

    if len(points):
        seg_class_img_draw.polygon(points, fill=fill_color)
        if seg_class_raw_img_draw:
            seg_class_raw_img_draw.polygon(points, fill=fill_color)

        # close the path
        points.append((points[0][0], points[0][1]))
        # draw border
        seg_class_img_draw.line(points, fill=BORDER_COLOR, width=BORDER_WIDTH, joint="curve")
        if seg_class_raw_img_draw:
            seg_class_raw_img_draw.line(points, fill=BORDER_COLOR, width=BORDER_WIDTH, joint="curve")

        # draw visualization_img
        if seg_class_visualization_img:
            seg_class_visualization_img_draw = ImageDraw.Draw(seg_class_visualization_img)
            seg_class_visualization_img_draw.line(
                points, fill=get_rgb_by_p_index(fill_color),
                width=BORDER_WIDTH,
                joint="curve"
            )
        pass


def draw_circle(seg_class_img, shape, seg_class_visualization_img=None, seg_class_raw_img=None):
    draw = ImageDraw.Draw(seg_class_img)
    points = shape['points']
    center_x = points[0][0]
    center_y = points[0][1]
    dx = center_x - points[1][0]
    dy = center_y - points[1][1]
    cr = math.sqrt(dx * dx + dy * dy)
    fill_color = get_label_color(shape['label'])
    draw.ellipse(
        xy=[(center_x - cr, center_y - cr), (center_x + cr, center_y + cr)],
        fill=fill_color, outline=BORDER_COLOR,
        width=BORDER_WIDTH
    )
    if seg_class_raw_img:
        seg_class_raw_img_draw = ImageDraw.Draw(seg_class_raw_img)
        seg_class_raw_img_draw.ellipse(
            xy=[(center_x - cr, center_y - cr), (center_x + cr, center_y + cr)],
            fill=fill_color, outline=BORDER_COLOR,
            width=BORDER_WIDTH
        )
    if seg_class_visualization_img:
        seg_class_visualization_img_draw = ImageDraw.Draw(seg_class_visualization_img)
        seg_class_visualization_img_draw.ellipse(
            xy=[(center_x - cr, center_y - cr), (center_x + cr, center_y + cr)],
            fill=None, outline=get_rgb_by_p_index(fill_color),
            width=BORDER_WIDTH
        )
        pass
    pass


def draw_rectangle(seg_class_img, shape, seg_class_visualization_img, seg_class_raw_img):
    seg_class_img_draw = ImageDraw.Draw(seg_class_img)
    seg_class_visualization_img_draw = ImageDraw.Draw(seg_class_visualization_img)
    seg_class_raw_img_draw = ImageDraw.Draw(seg_class_raw_img)
    points = translate_points(shape['points'])
    fill_color = get_label_color(shape['label'])
    seg_class_img_draw.rectangle(points, fill=fill_color, outline=BORDER_COLOR, width=BORDER_WIDTH)
    seg_class_raw_img_draw.rectangle(points, fill=fill_color, outline=BORDER_COLOR, width=BORDER_WIDTH)
    seg_class_visualization_img_draw.rectangle(
        points, fill=None, outline=get_rgb_by_p_index(fill_color),
        width=BORDER_WIDTH
    )
    pass


def gen_train_and_val_from_trainval(trainval_text_file_path):
    base_dir_path = os.path.dirname(trainval_text_file_path)
    fp = open(trainval_text_file_path, 'r')
    lines = fp.readlines()
    fp.close()

    # train.txt
    split_index = int(len(lines) * 0.95)
    fp = open(os.path.join(base_dir_path, TRAIN_TXT_FILE_NAME), 'w')
    fp.write("".join(lines[0:split_index]))
    fp.close()

    # var.txt
    fp = open(os.path.join(base_dir_path, VAL_TXT_FILE_NAME), 'w')
    fp.write("".join(lines[split_index:]))
    fp.close()


def gen_voc_dataset(labelme_dataset_input_dir, voc_dataset_output_dir):
    labelme_files = glob.glob(os.path.join(labelme_dataset_input_dir, "*.json"))
    trainval_txt_fp = open(get_trainval_text_file(voc_dataset_output_dir), 'w')
    for p in labelme_files:
        print(f"Transferring {p}")
        filename, ext = os.path.splitext(os.path.basename(p))
        f = open(p)
        fo = json.load(f)
        f.close()

        # load image data
        image_data_base64 = fo['imageData']
        image_data = base64.decodebytes(image_data_base64.encode("utf-8"))
        image_data_io = io.BytesIO(image_data)
        src_img = Image.open(image_data_io)

        # writing JPEGImages
        src_img.save(os.path.join(get_jpeg_images_dir(voc_dataset_output_dir), f"{filename}.jpg"), "JPEG")

        # draw shapes
        seg_class_img = Image.new(mode='P', size=src_img.size, color=0)
        seg_class_img.putpalette(COLOR_MAP)
        seg_class_raw_img = Image.new(mode='P', size=src_img.size)
        seg_class_visualization_img = Image.new(mode="RGBA", size=src_img.size)
        seg_class_visualization_img.paste(src_img)
        shapes = fo['shapes']
        for s in shapes:
            shape_type = s['shape_type']
            if shape_type == ShapeTypes.POLYGON:
                draw_polygon(
                    seg_class_img, s, seg_class_visualization_img=seg_class_visualization_img,
                    seg_class_raw_img=seg_class_raw_img
                )
            elif shape_type == ShapeTypes.CIRCLE:
                draw_circle(
                    seg_class_img, s, seg_class_visualization_img=seg_class_visualization_img,
                    seg_class_raw_img=seg_class_raw_img
                )
            elif shape_type == ShapeTypes.RECTANGLE:
                draw_rectangle(
                    seg_class_img, s, seg_class_visualization_img=seg_class_visualization_img,
                    seg_class_raw_img=seg_class_raw_img
                )
            else:
                raise NotImplementedError(f"Unsupported shape type {shape_type}")

        # writing SegmentationClass and SegmentationClassRaw
        seg_class_img.save(os.path.join(get_segmentation_class_dir(voc_dataset_output_dir), f"{filename}.png"), "PNG")
        seg_class_raw_img.save(
            os.path.join(get_segmentation_class_raw_dir(voc_dataset_output_dir), f"{filename}.png"),
            "PNG"
        )
        seg_class_visualization_img.save(
            os.path.join(get_segmentation_class_visualization_dir(voc_dataset_output_dir), f"{filename}.png"),
            "PNG"
        )

        # close the image io
        image_data_io.close()

        # write text files
        trainval_txt_fp.write(filename + "\n")
        pass

    trainval_txt_fp.close()
    # create train.txt, var.txt
    gen_train_and_val_from_trainval(get_trainval_text_file(voc_dataset_output_dir))

    # create labels.txt
    fp = open(get_labels_text_file(voc_dataset_output_dir), 'w')
    fp.write("\n".join(labels))
    fp.close()
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input_dir", type=str,  default="./", help="input annotated directory")
    if os.path.exists("VOCdevkit") == False:
        os.mkdir("VOCdevkit")
    if os.path.exists("VOCdevkit/VOC2012") == False:
        os.mkdir("VOCdevkit/VOC2012")
    parser.add_argument("--output_dir", type=str, default="./VOCdevkit/VOC2012", help="output dataset directory")

    args = parser.parse_args()
    check_to_make_output_dirs(args.output_dir)
    gen_voc_dataset(args.input_dir, args.output_dir)
    pass