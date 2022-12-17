import random
import pygame as pg
from PIL import Image, ImageEnhance
import pytesseract

# todo: распознавание цифр

# надо для работы pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

pg.init()
# размер экрана
sc = pg.display.set_mode((800, 600))
# цвет фона
WHITE = (255, 255, 255)
# кисть
RED = (255, 0, 0)
# фон
sc.fill(WHITE)
pg.display.update()

play = True
while play:
    for i in pg.event.get():
        # выход
        if i.type == pg.QUIT:
            pg.quit()
            play = False
        # если нажата пкм сохраняем картинку
        if i.type == pg.MOUSEBUTTONDOWN:
            if i.button == 3:
                pg.image.save(sc, 'picture.jpg')
                image = Image.open('picture.jpg')
                # улучшаем изображение
                enhancer = ImageEnhance.Contrast(image)
                img = enhancer.enhance(2)

                thresh = 200
                fn = lambda x: 255 if x > thresh else 0
                # Возвращает преобразованную копию этого изображения в чб.
                res = img.convert('L').point(fn, mode='1')
                res.save("bw.png", "png")
                # используем pytesseract
                text = pytesseract.image_to_string(res, lang='eng',
                                                   config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
                print(text)
            # на среднюю кнопку мыши очистить картинку
            if i.button == 2:
                sc.fill(WHITE)
                pg.display.update()
    if play:
        # airbrush = True
        # получаем позицию курсора
        cur = pg.mouse.get_pos()
        click = pg.mouse.get_pressed()
        # если нажимаем
        if click[0] == True:
            # рисуем
            pg.draw.circle(sc, RED, (cur[0], cur[1]), 10)
        pg.display.update()
