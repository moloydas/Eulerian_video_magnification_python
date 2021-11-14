#create_collapse_pyramids.py
import cv2

def create_laplacian_pyd(frame, levels):
    g_pyd = create_gaussian_pyd(frame, levels)
    pyd = [g_pyd[levels-1]]
    for i in range(levels-1, 0, -1):
        ge = cv2.pyrUp(g_pyd[i])
        L = cv2.subtract(g_pyd[i-1], ge)
        pyd.append(L)
    return pyd

def create_gaussian_pyd(frame, levels):
    img = frame.copy()
    pyd = [img]
    ge = img
    for i in range(levels):
        ge = cv2.pyrDown(ge)
        pyd.append(ge)
    return pyd

def collapse_laplacian_pyd(pyd, levels):
    lp = pyd[0]
    for i in range(1,levels):
        lp = cv2.pyrUp(lp)
        # lp = cv2.add(lp, pyd[i])
        lp = lp + pyd[i]
    return lp

if __name__ == '__main__':
    img = cv2.imread('img.jpg',1)
    pyd_lp = create_laplacian_pyd(img, 3)
    pyd_g = create_gaussian_pyd(img, 3)

    cv2.imshow('Org', img)

    # for i in range(len(pyd_lp)):
    #     cv2.imshow('Lp_'+str(i), pyd_lp[i])

    # for i in range(len(pyd_g)):
    #     cv2.imshow('Ge_'+str(i), pyd_g[i])

    img = collapse_laplacian_pyd(pyd_lp, 3)
    cv2.imshow('re constructed img', img)
    cv2.waitKey(0)
