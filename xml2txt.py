"""
XMLファイルを精度評価のためのtxtファイルへ変換する
"""

import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import cv2


def xml_to_txt(xml_path):
    #フォルダの中の全てのxmlファイルに対して
    for xml_file in glob.glob(xml_path + '/*.xml'): 
        tree = ET.parse(xml_file)
        root = tree.getroot()
        boxes = []
        filename = 'none'
        for member in root.findall('object'):
            filename = root.find('filename').text
            # クラス名，「右下・左下の座標」
            value = (
                     str(member[0].text,),
                     str(member[4][0].text),
                     str(member[4][1].text),
                     str(member[4][2].text),
                     str(member[4][3].text)
                     )
            boxes.append(value) # 1枚毎 複数のBBOXが保存

        # txtファイルに新規書き込み
        file = open("txt/" +filename.split(".")[0] + '.txt', 'w')
        for pp in boxes:
            my_list_str1 = ' '.join(pp) # gazoubboxの要素間に”半角スペース”入れて横並びに
            file.write(my_list_str1)
            file.write('\n')
        file.close()


def main():
    for directory in ['xml']: # xmlフォルダ内について探索
        xml_path = os.path.join(os.getcwd(), 'xml'.format(directory)) 
        xml_to_txt(xml_path)



if __name__ == '__main__':
    main()
