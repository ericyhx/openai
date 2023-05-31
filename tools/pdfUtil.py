from numbers import Number

from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
import fitz
import io
import os


endpoint="https://chatgpttest.cognitiveservices.azure.com/"
recognizer_key="8de81960bb394bb3bea4e277ba3eb9b4"
pages_per_embeddings=2
section_to_exclude = ['!footnote', '!pageHeader', '!pageFooter', '!pageNumber']


document_analysis_client = DocumentAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(recognizer_key))
output = io.StringIO()

def pdfToPng(pdf_path: str,imgs_path:str) -> Number:
    pdfDoc = fitz.open(pdf_path)
    png_counts=pdfDoc.page_count
    for pg in range(png_counts):
        page = pdfDoc[pg]
        rotate = int(0)
        # 每个尺寸的缩放系数为1.3，这将为我们生成分辨率提高2.6的图像。
        # 此处若是不做设置，默认图片大小为：792X612, dpi=96
        zoom_x = 1.33333333  # (1.33333333-->1056x816)   (2-->1584x1224)
        zoom_y = 1.33333333
        mat = fitz.Matrix(zoom_x, zoom_y).prerotate(rotate)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        if not os.path.exists(imgs_path):  # 判断存放图片的文件夹是否存在
            os.makedirs(imgs_path)  # 若图片文件夹不存在就创建

        pix.pil_save(imgs_path + '/' + 'images_%s.png' % pg)  # 将图片写入指定的文件夹内
    return png_counts

def parsePng(file_path: str) -> None:
    results = []
    with open(file_path, 'rb') as f:
        file_bytes = f.read()

    poller = document_analysis_client.begin_analyze_document("prebuilt-layout", file_bytes)
    layout = poller.result()

    for p in layout.paragraphs:
        page_number = p.bounding_regions[0].page_number
        output_file_id = int((page_number - 1) / pages_per_embeddings)

        if len(results) < output_file_id + 1:
            results.append('')

        if p.role not in section_to_exclude:
            results[output_file_id] += f"{p.content}\n"

    for t in layout.tables:
        page_number = t.bounding_regions[0].page_number
        output_file_id = int((page_number - 1) / pages_per_embeddings)

        if len(results) < output_file_id + 1:
            results.append('')
        previous_cell_row = 0
        rowcontent = '| '
        tablecontent = ''
        for c in t.cells:
            if c.row_index == previous_cell_row:
                rowcontent += c.content + " | "
            else:
                tablecontent += rowcontent + "\n"
                rowcontent = '|'
                rowcontent += c.content + " | "
                previous_cell_row += 1
        results[output_file_id] += f"{tablecontent}|"
    for r in results:
        output.write(r)
        output.write(' ')


def parsePdf2Txt(pdf_file_name:str,txt_file_name:str):
    temp_imgs_path=f"./imgs"
    imgs_size = pdfToPng(pdf_file_name, temp_imgs_path)
    for i in range(imgs_size):
        file_path = "{}/images_{}.png".format(temp_imgs_path, str(i))
        parsePng(file_path)
    fb = open(txt_file_name, mode="w", encoding="utf-8")
    fb.write(output.getvalue())
    del_files(temp_imgs_path)
def del_files(dir_path):
    if os.path.isfile(dir_path):
        try:
            os.remove(dir_path) # 这个可以删除单个文件，不能删除文件夹
        except BaseException as e:
            print(e)
    elif os.path.isdir(dir_path):
        file_lis = os.listdir(dir_path)
        for file_name in file_lis:
            tf = os.path.join(dir_path, file_name)
            del_files(tf)


if __name__ == '__main__':
   parsePdf2Txt('./files/pfs.pdf','./files/MAPF.txt')
