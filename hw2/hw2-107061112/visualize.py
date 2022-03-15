import numpy as np
import utils
import zipfile

if __name__ == "__main__":
    """
        We try to visualize the results of our generated npy files
        We will randomly select one image from each class.
        Since there are 9 classes, this function will show 
        nine "original + 5 generated images" pair.
    """
    classes = [
        'Center',
        'Donut',
        'Edge-Loc',
        'Edge-Ring',
        'Loc',
        'Near-full',
        'Random',
        'Scratch',
        'None'
    ]
    # A list to contain the original data
    orig_data = {}
    file = 'dataset/' + 'wafer.zip'
    # Open original datset
    with zipfile.ZipFile(file) as zf:
        for file_name in zf.namelist():
            dataset = np.load(file)
            orig_data[file_name] = dataset[file_name]
    gen_data = np.load('gen_data.npy')
    label = np.load('gen_label.npy')
    # retreive class info from original data
    temp = orig_data['label.npy'].ravel()
    # 9 classes, 9 images
    for i in range(9):
        # Randomly select an image from the ith class
        rand = np.random.choice(np.argwhere(temp == i).ravel(), 1)[0]
        # Mother image is the original image
        mother_img = orig_data['data.npy'][rand]
        decoded_img = []
        # Retreive the 5 generated images
        for j in range(5 * rand, 5 * rand + 5):
            decoded_img.append(gen_data[j])
        decoded_img = np.array(decoded_img)
        # Show the results on the fly
        utils.imshow(mother_img, decoded_img, str(classes[i]), vis=True)
