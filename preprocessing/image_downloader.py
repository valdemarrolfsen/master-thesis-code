import cv2
from skimage import io


x = 59.9256945
y = 10.71563044
zoom = 19
size = 400
map_type = "satellite"
base_url = "https://maps.googleapis.com/maps/api/staticmap"
api_key = "AIzaSyD3sIrrRRqyFNKrHeW58bplkmqHUXuG_Hg"

url = "{base_url}?center={x},{y}&zoom={zoom}&size={width}x{height}&maptype={type}&key={key}".format(
    base_url=base_url,
    x=x,
    y=y,
    zoom=zoom,
    width=size,
    height=size+40,  # So that we can crop out the google logo
    type=map_type,
    key=api_key
)

img = image = io.imread(url)
img = img[0:size, 0:size]

cv2.imshow('lalala', img)

if cv2.waitKey() & 0xff == 27:
    quit()
