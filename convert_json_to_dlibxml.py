import dlib
import cv2
import json

train_data_file = open('dataset/train_data.json', 'r')
train_data = json.load(train_data_file)
test_data_file = open('dataset/test_data.json', 'r')
test_data = json.load(test_data_file)

template_training = '''
<?xml version='1.0' encoding='ISO-8859-1'?>
<?xml-stylesheet type='text/xsl' href='image_metadata_stylesheet.xsl'?>
<dataset>
<name>Training faces</name>
<comment>HELEN Dataset
</comment>
<images>
'''

template_testing = '''
<?xml version='1.0' encoding='ISO-8859-1'?>
<?xml-stylesheet type='text/xsl' href='image_metadata_stylesheet.xsl'?>
<dataset>
<name>Testing faces</name>
<comment>HELEN Dataset
</comment>
<images>
'''

train_xml = template_training
test_xml = template_testing

for data in test_data:
    img_path = "dataset/helen/{0}.jpg".format(data["name"])
    annotations = data["annotations"]
    face_detector = dlib.get_frontal_face_detector()
    img = dlib.load_rgb_image(img_path)
    dets = face_detector(img, 1)
    if len(dets) == 0:
        print("test: " + img_path)
        continue
    test_xml += '''  <image file='{0}'>
'''.format(img_path)
    face_d = None
    dist = None
    for d in dets:
        if face_d is None or dist > abs(d.left() - annotations[0]["x"]):
            face_d = d
            dist = abs(d.left() - annotations[0]["x"])
    test_xml += '''    <box top='{0}' left='{1}' width='{2}' height='{3}'>
'''.format(face_d.top(), face_d.left(), face_d.width(), face_d.height())
    for i, annotation in enumerate(annotations):
        test_xml += '''      <part name='{0:02}' x='{1}' y='{2}'/>
'''.format(i, int(annotation["x"]), int(annotation["y"]))
    test_xml += '''    </box>
  </image>
'''

test_xml += '''</images>
</dataset>
'''

with open('test.xml', 'w') as test_xml_file:
    test_xml_file.write(test_xml)

for data in train_data:
    img_path = "dataset/helen/{0}.jpg".format(data["name"])
    annotations = data["annotations"]
    face_detector = dlib.get_frontal_face_detector()
    img = dlib.load_rgb_image(img_path)
    dets = face_detector(img, 1)
    if len(dets) == 0:
        print("train: " + img_path)
        continue
    train_xml += '''  <image file='{0}'>
'''.format(img_path)
    face_d = None
    dist = None
    for d in dets:
        if face_d is None or dist > abs(d.left() - annotations[0]["x"]):
            face_d = d
            dist = abs(d.left() - annotations[0]["x"])
    train_xml += '''    <box top='{0}' left='{1}' width='{2}' height='{3}'>
'''.format(face_d.top(), face_d.left(), face_d.width(), face_d.height())
    for i, annotation in enumerate(annotations):
        train_xml += '''      <part name='{0:02}' x='{1}' y='{2}'/>
'''.format(i, int(annotation["x"]), int(annotation["y"]))
    train_xml += '''    </box>
  </image>
'''

train_xml += '''</images>
</dataset>
'''

with open('train.xml', 'w') as train_xml_file:
    train_xml_file.write(train_xml)
