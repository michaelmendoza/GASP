import gdown
import os
import zipfile

import mapvbvd
import numpy as np

async def async_read_rawdata(filepath: str, datatype: str='image', doChaAverage: bool=True, doAveAverage: bool=True):
    return read_rawdata(filepath, datatype, doChaAverage, doAveAverage)

def read_rawdata(filepath: str, datatype: str='image', doChaAverage: bool=False, doChaSOSAverage: bool=False, doAveAverage: bool=True, use3dSlices: bool=False):
    """ Reads rawdata files and returns NodeDataset. """
    
    if doChaSOSAverage:
        doChaAverage = False

    twixObj = mapvbvd.mapVBVD(filepath)
    sqzDims = twixObj.image.sqzDims    
    twixObj.image.squeeze = True

    data = twixObj.image['']
    # Move Lin be first index
    linIndex = sqzDims.index('Lin')
    data = np.moveaxis(data, linIndex, 0)
    sqzDims.insert(0, sqzDims.pop(linIndex))

    # Handle averages dimension 
    if doAveAverage and 'Ave' in sqzDims:
        chaIndex = sqzDims.index('Ave')
        data = np.mean(data, axis=chaIndex)
        sqzDims.pop(chaIndex)
                
    # Handle 3d data
    if 'Par' in sqzDims:
        sliceIndex = sqzDims.index('Par')
        data = np.moveaxis(data, sliceIndex, 0)
        sqzDims.insert(0, sqzDims.pop(sliceIndex))
        is3D = True
    else:
        is3D = False

    # Handle fft if required
    if datatype == 'image':
        if is3D:
            data = np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(data, axes=(0, 1, 2))))
        else:
            data = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(data, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))
    else:  # datatype is kspace
        pass

    # Handle coils dimension - Options for averaging coil data
    if (doChaAverage or doChaSOSAverage) and 'Cha' in sqzDims:
        chaIndex = sqzDims.index('Cha')

        if doChaAverage:
            data = np.mean(data, axis=chaIndex)
        elif doChaSOSAverage:
            data = np.sqrt(np.sum(data**2, axis=chaIndex))
 
        sqzDims.pop(chaIndex)

    # Handle 3D data 
    if 'Sli' in sqzDims:
         # Handle depth - Make data[depth, height, width, ...] Sli = depth
        sliceIndex = sqzDims.index('Sli')
        data = np.moveaxis(data, sliceIndex, 0)
        sqzDims.insert(0, sqzDims.pop(sliceIndex))
    elif use3dSlices: 
        # Make data[depth, height, width, ...] Sli = depth
        # Set depth to 1 if 2d image
        sqzDims.insert(0, 'Sli')
        data = data[np.newaxis, ...]

    mmin = float(np.nanmin(np.abs(data)))
    mmax = float(np.nanmax(np.abs(data)))
    isComplex = np.iscomplexobj(data)

    header = twixObj.hdr
    return {'data': data, 'dims': sqzDims, 'shape': data.shape, 'min': mmin, 'max': mmax, 'isComplex': isComplex}


def download_data(url: str, dataname: str, path: str= os.getcwd()):
    
    # Checks if data folder exists
    targetdir = os.path.join(path, 'data')  
    if not os.path.exists(targetdir):
        os.mkdir(targetdir)

    # Checks if data exists
    filepath = os.path.join(path, 'data', dataname)
    fileExists = os.path.exists(filepath)
    if fileExists:
        print(f'Data: {dataname} data exists')
        return

    # Downloads data
    print('Downloading files ...')
    output = filepath + '.zip'
    gdown.download(url, quiet=False, output=output, fuzzy=True)
    print('Download complete.')

    # Extracts data from .zip 
    print('Extracting files ...')
    with zipfile.ZipFile(output,"r") as zip_ref:
        zip_ref.extractall(targetdir)
        print('Extract complete.')
        print(f'Data located at: ${filepath}')

    # Remove downloaded zip file 
    os.remove(output)
