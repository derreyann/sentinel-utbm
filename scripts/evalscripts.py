evalscript_ndvi = """
//VERSION=3
function setup() {
  return {
    input: [{
      bands: [
        "B04",
        "B08",
        "dataMask"
      ]
    }],
    output: {
      bands: 4
    }
  }
}

function evaluatePixel(sample) {
    let val = (sample.B08 - sample.B04) / (sample.B08 + sample.B04);
    let imgVals = null;
    
    if (val<-1.1) imgVals = [0,0,0];
    else if (val<-0.2) imgVals = [0.75,0.75,0.75];
    else if (val<-0.1) imgVals = [0.86,0.86,0.86];
    else if (val<0) imgVals = [1,1,0.88];
    else if (val<0.025) imgVals = [1,0.98,0.8];
    else if (val<0.05) imgVals = [0.93,0.91,0.71];
    else if (val<0.075) imgVals = [0.87,0.85,0.61];
    else if (val<0.1) imgVals = [0.8,0.78,0.51];
    else if (val<0.125) imgVals = [0.74,0.72,0.42];
    else if (val<0.15) imgVals = [0.69,0.76,0.38];
    else if (val<0.175) imgVals = [0.64,0.8,0.35];
    else if (val<0.2) imgVals = [0.57,0.75,0.32];
    else if (val<0.25) imgVals = [0.5,0.7,0.28];
    else if (val<0.3) imgVals = [0.44,0.64,0.25];
    else if (val<0.35) imgVals = [0.38,0.59,0.21];
    else if (val<0.4) imgVals = [0.31,0.54,0.18];
    else if (val<0.45) imgVals = [0.25,0.49,0.14];
    else if (val<0.5) imgVals = [0.19,0.43,0.11];
    else if (val<0.55) imgVals = [0.13,0.38,0.07];
    else if (val<0.6) imgVals = [0.06,0.33,0.04];
    else imgVals = [0,0.27,0];
    
    imgVals.push(sample.dataMask)
    
    return imgVals
}
"""

evalscript_ndvi2 = """
//VERSION=3
let ndvi = (B08 - B04) / (B08 + B04);

return colorBlend(ndvi,
   [-0.2, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 ],
   [[0, 0, 0,dataMask],							   //  < -.2 = #000000 (black)
    [165/255,0,38/255,dataMask],        //  -> 0 = #a50026
    [215/255,48/255,39/255,dataMask],   //  -> .1 = #d73027
    [244/255,109/255,67/255,dataMask],  //  -> .2 = #f46d43
    [253/255,174/255,97/255,dataMask],  //  -> .3 = #fdae61
    [254/255,224/255,139/255,dataMask], //  -> .4 = #fee08b
    [255/255,255/255,191/255,dataMask], //  -> .5 = #ffffbf
    [217/255,239/255,139/255,dataMask], //  -> .6 = #d9ef8b
    [166/255,217/255,106/255,dataMask], //  -> .7 = #a6d96a
    [102/255,189/255,99/255,dataMask],  //  -> .8 = #66bd63
    [26/255,152/255,80/255,dataMask],   //  -> .9 = #1a9850
    [0,104/255,55/255,dataMask]         //  -> 1.0 = #006837
   ]);
"""


evalscript_CloudFree_Composite = """
//VERSION=3
function setup() {
    return {
        input: ["B08", "B04", "B03", "SCL"],
        output: { bands: 4 }
    };
}

function evaluatePixel(sample) {
    if (sample.SCL == 3 || sample.SCL == 8 || sample.SCL == 9 || sample.SCL == 10) {
        // Return transparent for clouds, shadows, and snow
        return [0, 0, 0, 0];
    }
    return [sample.B08, sample.B04, sample.B03, 1];
}
"""
