package org.deeplearning4j.examples.dataExamples;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.CropImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.MultiImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Random;


/**
 * Created By: Fuxing Loh
 * Date: 13/10/2016
 * Time: 10:55 PM
 * Project: deeplearning4j-examples-parent
 */
public class FinnDataSet {

    protected static final Logger log = LoggerFactory.getLogger(ImagePipelineExample.class);

    //Images are of format given by allowedExtension -
    protected static final String[] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
    public static final Random randNumGen = new Random(12345);

    protected static int height = 100;
    protected static int width = 100;
    protected static int channels = 3;
    protected static int outputNum = 3;

    private DataSetIterator train, test;

    public FinnDataSet() throws IOException {
        File parentDir = new File("C:\\Users\\Fuxin\\Documents\\", "FinnPipeline\\");
        //Files in directories under the parent dir that have "allowed extensions" plit needs a random number generator for reproducibility when splitting the files into train and test
        FileSplit filesInDir = new FileSplit(parentDir, allowedExtensions, randNumGen);

        //You do not have to manually specify labels. This class (instantiated as below) will
        //parse the parent dir and use the name of the subdirectories as label/class names
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        //The balanced path filter gives you fine tune control of the min/max cases to load for each class
        //Below is a bare bones version. Refer to javadocs for details
        BalancedPathFilter pathFilter = new BalancedPathFilter(randNumGen, allowedExtensions, labelMaker);

        //Split the image files into train and test. Specify the train test split as 80%,20%
        InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, 80, 20);
        InputSplit trainData = filesInDirSplit[0];
        InputSplit testData = filesInDirSplit[1];

        //Specifying a new record reader with the height and width you want the images to be resized to.
        //Note that the images in this example are all of different size
        //They will all be resized to the height and width specified below
        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);

        //Often there is a need to transforming images to artificially increase the size of the dataset
        //DataVec has built in powerful features from OpenCV
        //You can chain transformations as shown below, write your own classes that will say detect a face and crop to size
        /*ImageTransform transform = new MultiImageTransform(randNumGen,
            new CropImageTransform(10), new FlipImageTransform(),
            new ScaleImageTransform(10), new WarpImageTransform(10));
            */

        //You can use the ShowImageTransform to view your images
        //Code below gives you a look before and after, for a side by side comparison
        ImageTransform transform = new MultiImageTransform(randNumGen, new CropImageTransform(100));

        recordReader.initialize(trainData, transform);
        train = new RecordReaderDataSetIterator(recordReader, 64, 1, outputNum);

        recordReader.initialize(testData, transform);
        test = new RecordReaderDataSetIterator(recordReader, 64, 1, outputNum);
    }

    public DataSetIterator getTrain() {
        return train;
    }

    public DataSetIterator getTest() {
        return test;
    }
}
