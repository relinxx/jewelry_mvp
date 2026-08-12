import os

import cv2
import fitz
import numpy as np


PDF_PATH = "rings_bangles.pdf"
OUTPUT_DIR = "output"

RENDER_SCALE = 3.0


# ------------------------------------------------------------
# Configuration for THIS catalog
# ------------------------------------------------------------
#
# rows/cols:
#   Rings   -> 6 rows x 5 columns
#   Bangles -> 3 rows x 3 columns
#
# first_id:
#   first K.M. number on that page
#
# rotate:
#   page 12 in this PDF is sideways
#
# Page 9 duplicates the K.M. 679-708 page,
# so we skip it.
# ------------------------------------------------------------

PAGE_CONFIG = {
    1: {
        "category": "rings",
        "first_id": 409,
        "rows": 6,
        "cols": 5,
    },
    2: {
        "category": "rings",
        "first_id": 439,
        "rows": 6,
        "cols": 5,
    },
    3: {
        "category": "rings",
        "first_id": 529,
        "rows": 6,
        "cols": 5,
    },
    4: {
        "category": "rings",
        "first_id": 559,
        "rows": 6,
        "cols": 5,
    },
    5: {
        "category": "rings",
        "first_id": 589,
        "rows": 6,
        "cols": 5,
    },
    6: {
        "category": "rings",
        "first_id": 619,
        "rows": 6,
        "cols": 5,
    },
    7: {
        "category": "rings",
        "first_id": 649,
        "rows": 6,
        "cols": 5,
    },
    8: {
        "category": "rings",
        "first_id": 679,
        "rows": 6,
        "cols": 5,
    },

    # Page 9 duplicates page 8, therefore deliberately skipped.

    10: {
        "category": "bangles",
        "first_id": 1012,
        "rows": 3,
        "cols": 3,
    },
    11: {
        "category": "bangles",
        "first_id": 1021,
        "rows": 3,
        "cols": 3,
    },
    12: {
        "category": "bangles",
        "first_id": 1048,
        "rows": 3,
        "cols": 3,
        "rotate": "ccw",
    },
    13: {
        "category": "bangles",
        "first_id": 1057,
        "rows": 3,
        "cols": 3,
    },
}


# ------------------------------------------------------------
# Create directories
# ------------------------------------------------------------

def create_output_directories():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    os.makedirs(
        os.path.join(OUTPUT_DIR, "rings"),
        exist_ok=True,
    )

    os.makedirs(
        os.path.join(OUTPUT_DIR, "bangles"),
        exist_ok=True,
    )

    os.makedirs(
        os.path.join(OUTPUT_DIR, "debug"),
        exist_ok=True,
    )


# ------------------------------------------------------------
# Convert PDF page to OpenCV image
# ------------------------------------------------------------

def render_pdf_page(page):
    matrix = fitz.Matrix(
        RENDER_SCALE,
        RENDER_SCALE,
    )

    pixmap = page.get_pixmap(
        matrix=matrix,
        alpha=False,
    )

    image = np.frombuffer(
        pixmap.samples,
        dtype=np.uint8,
    )

    image = image.reshape(
        pixmap.height,
        pixmap.width,
        pixmap.n,
    )

    # PyMuPDF -> RGB
    # OpenCV -> BGR
    image = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR,
    )

    return image


# ------------------------------------------------------------
# Rotate page if necessary
# ------------------------------------------------------------

def rotate_image(image, direction):
    if direction == "ccw":
        return cv2.rotate(
            image,
            cv2.ROTATE_90_COUNTERCLOCKWISE,
        )

    if direction == "cw":
        return cv2.rotate(
            image,
            cv2.ROTATE_90_CLOCKWISE,
        )

    return image


# ------------------------------------------------------------
# Define useful part of page
# ------------------------------------------------------------

def get_page_roi(image, category):
    height, width = image.shape[:2]

    if category == "rings":

        # Slightly remove page edges / branding.
        left = int(width * 0.03)
        right = int(width * 0.97)

        top = int(height * 0.01)
        bottom = int(height * 0.94)

    else:

        # Bangle pages use almost the entire page.
        left = int(width * 0.01)
        right = int(width * 0.99)

        top = int(height * 0.005)
        bottom = int(height * 0.995)

    roi = image[
        top:bottom,
        left:right
    ]

    return roi


# ------------------------------------------------------------
# Find jewelry pixels
# ------------------------------------------------------------

def create_gold_mask(cell):
    """
    Jewelry in this catalog is mostly gold/orange.

    HSV makes it much easier to separate:
        gold jewelry
    from
        white background
        black text
        grey display rods
    """

    hsv = cv2.cvtColor(
        cell,
        cv2.COLOR_BGR2HSV,
    )

    # Gold / orange color range
    lower_gold = np.array(
        [2, 60, 60]
    )

    upper_gold = np.array(
        [45, 255, 255]
    )

    mask = cv2.inRange(
        hsv,
        lower_gold,
        upper_gold,
    )

    # Remove small noise
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5),
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1,
    )

    return mask


# ------------------------------------------------------------
# Find useful components
# ------------------------------------------------------------

def get_components(mask):
    number_of_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )
    )

    components = []

    cell_area = (
        mask.shape[0]
        * mask.shape[1]
    )

    # Relative threshold so it still works
    # if rendering resolution changes.
    minimum_area = cell_area * 0.001

    for index in range(
        1,
        number_of_labels
    ):
        x = stats[
            index,
            cv2.CC_STAT_LEFT
        ]

        y = stats[
            index,
            cv2.CC_STAT_TOP
        ]

        width = stats[
            index,
            cv2.CC_STAT_WIDTH
        ]

        height = stats[
            index,
            cv2.CC_STAT_HEIGHT
        ]

        area = stats[
            index,
            cv2.CC_STAT_AREA
        ]

        if area < minimum_area:
            continue

        if width < 5:
            continue

        if height < 5:
            continue

        components.append(
            {
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "area": area,
            }
        )

    return components


# ------------------------------------------------------------
# Crop ring
# ------------------------------------------------------------

def crop_ring(cell):
    mask = create_gold_mask(cell)

    components = get_components(mask)

    if len(components) == 0:
        print(
            "WARNING: No ring detected."
        )

        return cell

    # Largest connected component should
    # normally belong to the ring.
    largest = max(
        components,
        key=lambda component:
        component["area"],
    )

    largest_center_x = (
        largest["x"]
        + largest["width"] / 2
    )

    largest_center_y = (
        largest["y"]
        + largest["height"] / 2
    )

    useful_components = []

    cell_height, cell_width = (
        cell.shape[:2]
    )

    # Include components close to the main
    # ring component.
    #
    # This is useful for stones or decorative
    # pieces that aren't physically connected
    # in the mask.
    for component in components:

        center_x = (
            component["x"]
            + component["width"] / 2
        )

        center_y = (
            component["y"]
            + component["height"] / 2
        )

        horizontal_distance = abs(
            center_x
            - largest_center_x
        )

        vertical_distance = abs(
            center_y
            - largest_center_y
        )

        if (
            horizontal_distance
            < cell_width * 0.40
            and
            vertical_distance
            < cell_height * 0.40
        ):
            useful_components.append(
                component
            )

    return crop_components(
        cell,
        useful_components,
        horizontal_padding=18,
        vertical_padding=12,
    )


# ------------------------------------------------------------
# Crop bangle set
# ------------------------------------------------------------

def crop_bangle(cell):
    mask = create_gold_mask(cell)

    components = get_components(mask)

    if len(components) == 0:
        print(
            "WARNING: No bangle detected."
        )

        return cell

    useful_components = []

    cell_height, cell_width = (
        cell.shape[:2]
    )

    for component in components:
        center_x = (
            component["x"]
            + component["width"] / 2
        )

        center_y = (
            component["y"]
            + component["height"] / 2
        )

        # Ignore strange artifacts right at
        # extreme page edges.
        if center_x < cell_width * 0.02:
            continue

        if center_x > cell_width * 0.98:
            continue

        # Tiny components extremely near the bottom
        # are usually catalog text rather than jewelry.
        if (
            center_y > cell_height * 0.93
            and
            component["height"]
            < cell_height * 0.10
        ):
            continue

        useful_components.append(
            component
        )

    if len(useful_components) == 0:
        return cell

    return crop_components(
        cell,
        useful_components,
        horizontal_padding=10,
        vertical_padding=7,
    )


# ------------------------------------------------------------
# Crop bounding rectangle
# ------------------------------------------------------------

def crop_components(
    image,
    components,
    horizontal_padding=10,
    vertical_padding=10,
):
    if len(components) == 0:
        return image

    x1 = min(
        component["x"]
        for component
        in components
    )

    y1 = min(
        component["y"]
        for component
        in components
    )

    x2 = max(
        component["x"]
        + component["width"]
        for component
        in components
    )

    y2 = max(
        component["y"]
        + component["height"]
        for component
        in components
    )

    x1 -= horizontal_padding
    x2 += horizontal_padding

    y1 -= vertical_padding
    y2 += vertical_padding

    height, width = image.shape[:2]

    x1 = max(
        0,
        x1,
    )

    y1 = max(
        0,
        y1,
    )

    x2 = min(
        width,
        x2,
    )

    y2 = min(
        height,
        y2,
    )

    cropped = image[
        y1:y2,
        x1:x2
    ]

    return cropped


# ------------------------------------------------------------
# Make image square
# ------------------------------------------------------------

def make_square(image, size=512):
    """
    Adds white padding around the item.

    This is useful later when generating
    embeddings because every image has a
    consistent shape.
    """

    height, width = image.shape[:2]

    square_size = max(
        height,
        width,
    )

    square = np.full(
        (
            square_size,
            square_size,
            3,
        ),
        255,
        dtype=np.uint8,
    )

    x_offset = (
        square_size - width
    ) // 2

    y_offset = (
        square_size - height
    ) // 2

    square[
        y_offset:
        y_offset + height,

        x_offset:
        x_offset + width
    ] = image

    square = cv2.resize(
        square,
        (size, size),
        interpolation=cv2.INTER_AREA,
    )

    return square


# ------------------------------------------------------------
# Draw debug grid
# ------------------------------------------------------------

def save_debug_grid(
    roi,
    page_number,
    rows,
    cols,
):
    debug_image = roi.copy()

    height, width = (
        debug_image.shape[:2]
    )

    for row in range(
        1,
        rows
    ):
        y = int(
            row
            * height
            / rows
        )

        cv2.line(
            debug_image,
            (0, y),
            (width, y),
            (0, 0, 255),
            3,
        )

    for column in range(
        1,
        cols
    ):
        x = int(
            column
            * width
            / cols
        )

        cv2.line(
            debug_image,
            (x, 0),
            (x, height),
            (0, 0, 255),
            3,
        )

    path = os.path.join(
        OUTPUT_DIR,
        "debug",
        f"page_{page_number:02d}_grid.jpg",
    )

    cv2.imwrite(
        path,
        debug_image,
    )


# ------------------------------------------------------------
# Process one page
# ------------------------------------------------------------

def process_page(
    image,
    page_number,
    config,
):
    category = config["category"]

    rows = config["rows"]
    cols = config["cols"]

    first_id = config["first_id"]

    rotate = config.get(
        "rotate"
    )

    if rotate is not None:
        image = rotate_image(
            image,
            rotate,
        )

    roi = get_page_roi(
        image,
        category,
    )

    save_debug_grid(
        roi,
        page_number,
        rows,
        cols,
    )

    roi_height, roi_width = (
        roi.shape[:2]
    )

    product_index = 0

    for row in range(rows):

        y1 = int(
            row
            * roi_height
            / rows
        )

        y2 = int(
            (row + 1)
            * roi_height
            / rows
        )

        for column in range(cols):

            x1 = int(
                column
                * roi_width
                / cols
            )

            x2 = int(
                (column + 1)
                * roi_width
                / cols
            )

            cell = roi[
                y1:y2,
                x1:x2
            ]

            product_id = (
                first_id
                + product_index
            )

            if category == "rings":
                item = crop_ring(
                    cell
                )

            else:
                item = crop_bangle(
                    cell
                )

            # Consistent white 512x512 output.
            item = make_square(
                item,
                size=512,
            )

            filename = (
                f"KM_{product_id}.jpg"
            )

            path = os.path.join(
                OUTPUT_DIR,
                category,
                filename,
            )

            cv2.imwrite(
                path,
                item,
            )

            print(
                f"Saved {path}"
            )

            product_index += 1


# ------------------------------------------------------------
# Main program
# ------------------------------------------------------------

def main():
    create_output_directories()

    pdf = fitz.open(
        PDF_PATH
    )

    print(
        f"PDF pages: {len(pdf)}"
    )

    for page_number in range(
        1,
        len(pdf) + 1
    ):

        if page_number not in PAGE_CONFIG:
            print(
                f"Skipping page "
                f"{page_number}"
            )

            continue

        print(
            "\n"
            + "=" * 50
        )

        print(
            f"Processing page "
            f"{page_number}"
        )

        page = pdf[
            page_number - 1
        ]

        image = render_pdf_page(
            page
        )

        config = PAGE_CONFIG[
            page_number
        ]

        process_page(
            image,
            page_number,
            config,
        )

    pdf.close()

    print(
        "\nExtraction finished."
    )

    print(
        f"Check the '{OUTPUT_DIR}' "
        f"folder."
    )


if __name__ == "__main__":
    main()