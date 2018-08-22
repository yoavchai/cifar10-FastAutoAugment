from alu_transforms import *


class sub0(object):
    def __init__(self):
        self.brightness = brightness()
        self.brightness.mag_scale_value = 2
        self.brightness.curr_p = 0.8

        self.cutout = cutout()
        self.cutout.mag_scale_value = 3
        self.cutout.curr_p = 1

    def aug(self,img):
        img = self.brightness.do_aug(img, self.brightness.curr_p, self.brightness.mag_scale_value)
        img = self.cutout.do_aug(img, self.cutout.curr_p, self.cutout.mag_scale_value)

        return img

    @staticmethod
    def my_name():
        return 'sub0'

class sub1(object):
    def __init__(self):
        self.color = color()
        self.color.mag_scale_value = 8
        self.color.curr_p = 0.4

        self.brightness = brightness()
        self.brightness.mag_scale_value = 2
        self.brightness.curr_p = 0.8



    def aug(self,img):
        img = self.color.do_aug(img, self.color.curr_p, self.color.mag_scale_value)
        img = self.brightness.do_aug(img, self.brightness.curr_p, self.brightness.mag_scale_value)

        return img

    @staticmethod
    def my_name():
        return 'sub1'

class sub2(object):
    def __init__(self):
        self.color = color()
        self.color.mag_scale_value = 5
        self.color.curr_p = 0.2

        self.cutout = cutout()
        self.cutout.mag_scale_value = 3
        self.cutout.curr_p = 1

    def aug(self,img):
        img = self.color.do_aug(img, self.color.curr_p, self.color.mag_scale_value)
        img = self.cutout.do_aug(img, self.cutout.curr_p, self.cutout.mag_scale_value)

        return img

    @staticmethod
    def my_name():
        return 'sub2'

class sub3(object):
    def __init__(self):
        self.cutout0 = color()
        self.cutout0.mag_scale_value = 2
        self.cutout0.curr_p = 0.3

        self.cutout1 = cutout()
        self.cutout1.mag_scale_value = 3
        self.cutout1.curr_p = 1

    def aug(self,img):
        img = self.cutout0.do_aug(img, self.cutout0.curr_p, self.cutout0.mag_scale_value)
        img = self.cutout1.do_aug(img, self.cutout1.curr_p, self.cutout1.mag_scale_value)

        return img

    @staticmethod
    def my_name():
        return 'sub3'


class sub4(object):
    def __init__(self):
        self.sharpness = sharpness()
        self.sharpness.mag_scale_value = 9
        self.sharpness.curr_p =0.5

        self.cutout = cutout()
        self.cutout.mag_scale_value = 3
        self.cutout.curr_p = 1

    def aug(self,img):
        img = self.sharpness.do_aug(img, self.sharpness.curr_p, self.sharpness.mag_scale_value)
        img = self.cutout.do_aug(img, self.cutout.curr_p, self.cutout.mag_scale_value)

        return img

    @staticmethod
    def my_name():
        return 'sub4'

class sub5(object):

    def __init__(self):
        self.contrast = contrast()
        self.contrast.mag_scale_value = 9
        self.contrast.curr_p =0.1

        self.cutout = cutout()
        self.cutout.mag_scale_value = 3
        self.cutout.curr_p = 1

    def aug(self,img):
        img = self.contrast.do_aug(img, self.contrast.curr_p, self.contrast.mag_scale_value)
        img = self.cutout.do_aug(img, self.cutout.curr_p, self.cutout.mag_scale_value)

        return img

    @staticmethod
    def my_name():
        return 'sub5'

class sub6(object):

    def __init__(self):
        self.sharpness = sharpness()
        self.sharpness.mag_scale_value = 7
        self.sharpness.curr_p =0.2

        self.color = color()
        self.color.mag_scale_value = 5
        self.color.curr_p = 0.2

    def aug(self,img):
        img = self.sharpness.do_aug(img, self.sharpness.curr_p, self.sharpness.mag_scale_value)
        img = self.color.do_aug(img, self.color.curr_p, self.color.mag_scale_value)

        return img

    @staticmethod
    def my_name():
        return 'sub6'

class sub7(object):
    def __init__(self):
        self.color = color()
        self.color.mag_scale_value = 5
        self.color.curr_p =0.2

        self.autocontrast = autocontrast()
        self.autocontrast.mag_scale_value = 0
        self.autocontrast.curr_p = 0.5

    def aug(self,img):
        img = self.color.do_aug(img, self.color.curr_p, self.color.mag_scale_value)
        img = self.autocontrast.do_aug(img, self.autocontrast.curr_p, self.autocontrast.mag_scale_value)

        return img

    @staticmethod
    def my_name():
        return 'sub7'

class sub8(object):
    def __init__(self):
        self.autocontrast = autocontrast()
        self.autocontrast.mag_scale_value = 0
        self.autocontrast.curr_p =0.5

        self.cutout = cutout()
        self.cutout.mag_scale_value = 2
        self.cutout.curr_p = 0.3

    def aug(self,img):
        img = self.autocontrast.do_aug(img, self.autocontrast.curr_p, self.autocontrast.mag_scale_value)
        img = self.cutout.do_aug(img, self.cutout.curr_p, self.cutout.mag_scale_value)

        return img

    @staticmethod
    def my_name():
        return 'sub8'

