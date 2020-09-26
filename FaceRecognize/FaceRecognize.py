#-*- coding:utf-8 -*
import cv2
import os
#import pathlib

def faceRecognize():
    path=os.path.dirname(os.path.abspath(__file__))
    #フォルダ名の取得(abspathで絶対パスを取得。どこにフォルダがあっても動くようにする)
    #path=str(pathlib.Path().resolve())
    #path= Path(__file__).parent.absolute().joinpath('haarcascade_frontalface_default.xml')

    face_path=(path+"\haarcascade_frontalface_default.xml")
    face_cascade=cv2.CascadeClassifier(face_path)
    
    # 分類子（めちゃくちゃ簡単にいうと、それぞれの特徴的なやつ）の読み込み
    #→あらかじめ学習されている？データをもとに解析する

    eye_path=(path+"\haarcascade_eye.xml")
    eye_cascade=cv2.CascadeClassifier(eye_path)
    #上と同じ

    red = (0, 0, 255)
    green=(0,255,0)
    cap = cv2.VideoCapture(0)
    #引数に0でカメラを指定

    while True:
        #カメラが開いている限り
        ret, frame = cap.read()
        #今のカメラの画像を読み込む(frameと名付ける)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #グレースケール(なんで？)→色を単純化した方が処理が早くなる
        # →そのためグレーで位置を取得、その座標を普通の画像に写すって作業をする
        face_rect = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.4, minNeighbors=4)
                    # detectMultiScare(画像、縮小量(顔の認知に関係する。少ないと誤認、大きいと認識しない)、近傍矩形（長方形の大きさ？）)
        #認識するものの調整。(リストになるっぽい)
        # →（多分だけど）位置情報を認識した数だけ追加してるっぽい→見つからないと、（）を受け取る
        #eye_rect = eye_cascade.detectMultiScale(gray_scare, scaleFactor=1.5, minNeighbors=2)
        

        for x, y, w, h in face_rect:
                #rectには四つの配列(囲む範囲？)が一つに格納される。
            cv2.rectangle(frame, (x, y), (x+w, y+h), red)
                #顔を囲む長方形を作成、書く(数だけ繰り返す)
                #片側の頂点の座標を取得、計算して反対側も取得
            face_gray = gray_frame[y:y + h, x:x + w]
            face_color = frame[y:y + h, x:x + w]
            eye_rect = eye_cascade.detectMultiScale(face_gray, scaleFactor=1.3)

            for x,y,w,h in eye_rect:
                cv2.rectangle(face_color, (x, y), (x+w, y+h),green )
                    #同じようにフレームから取得する
        cv2.imshow("capture", frame)
        # フレーム(今カメラに写ってるやつ)を表示（図形描写を加えて）
        if cv2.waitKey(1) & 0xFF == ord('q'):
            #引数を0にすると入力するまで動かなくなるので1秒ずつ。
            # また0xffはnumlockなどに対応するのに必要
            break
            #ループ抜けて、下の処理へ
    cap.release()
    #キャプチャをやめる
    cv2.destroyAllWindows()
    #すべてのウィンドウを閉じる。現状はなくてもかわんない

if __name__ == "__main__":
    faceRecognize()
    