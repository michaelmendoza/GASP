import numpy as np
import mapvbvd

async def async_read_rawdata(filepath, datatype='image', doChaAverage = True, doAveAverage = True):
    return read_rawdata(filepath, datatype, doChaAverage, doAveAverage)

def read_rawdata(filepath, datatype='image', doChaAverage = True, doAveAverage = True):
    ''' Reads rawdata files and returns NodeDataset '''

    twixObj = mapvbvd.mapVBVD(filepath)
    sqzDims = twixObj.image.sqzDims    
    twixObj.image.squeeze = True

    data = twixObj.image['']
    # Move Lin be first index
    linIndex = sqzDims.index('Lin')
    data = np.moveaxis(data, linIndex, 0)
    sqzDims.insert(0, sqzDims.pop(linIndex))

    if doChaAverage and 'Cha' in sqzDims:
        chaIndex = sqzDims.index('Cha')
        data = np.mean(data, axis=chaIndex)
        sqzDims.pop(chaIndex)

    if doAveAverage and 'Ave' in sqzDims:
        chaIndex = sqzDims.index('Ave')
        data = np.mean(data, axis=chaIndex)
        sqzDims.pop(chaIndex)
                
    if datatype == 'image':
        data = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(data, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))
    else: # datatype is kspace
        pass

    if 'Sli' in sqzDims:
        sliceIndex = sqzDims.index('Sli')
        data = np.moveaxis(data, sliceIndex, 0)
        sqzDims.insert(0, sqzDims.pop(sliceIndex))

    return { 'data':data, 'dims':sqzDims, 'shape':data.shape }