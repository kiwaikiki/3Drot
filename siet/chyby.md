my_train.py
- num workers bolo 0 a zralo nekonecno pamate(zabilo ma po 9.epoche) -> 4 bolo fajn
- neukladala som natrenovany model 

my_dataset.py
- csv bolo offsetnute o 1, nechapem
- vracala som v dataste zly index (+ 10 problemov s reindexovanim) -> index2pic_id dict

my_infer/eval.py
- nenacitala spravny model na infer (lebo som pouzivala ich stare args ktore nemali nieco spravne)

my_loss.py
- vsade v losse np. namiesto torch. - tam kde som chcel este pouzivat loss
- tam kde numpy - cpu.detach
- =+ robi problemy (setting an array element with a sequence.. nejaky zly gradient potom) -> x = x + 5
- v losse sa vyhodnocuje cely batch -> prerobit na velko vektorove operacie
- ulozena matica bola double, ale vyrobena transform bol float -> pretypovat




PROBLEMY RANDOM:
- bpy nejde v conde
