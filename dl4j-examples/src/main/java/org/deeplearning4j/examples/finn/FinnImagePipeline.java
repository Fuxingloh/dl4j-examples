package org.deeplearning4j.examples.finn;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
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
 * Time: 11:24 PM
 * Project: deeplearning4j-examples-parent
 */
public class FinnImagePipeline {

    protected static final Logger log = LoggerFactory.getLogger(FinnImagePipeline.class);

    private static final String[] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
    private static final Random randNumGen = new Random(12345);

    // Image Settings
    protected static int height = 100;
    protected static int width = 100;
    protected static int channels = 3;
    protected static int outputNum = 3;

    private ParentPathLabelGenerator labelMaker;
    private InputSplit trainData;
    private InputSplit testData;

    public FinnImagePipeline() throws IOException {
        File parentDir = new File("C:\\Users\\Fuxin\\Google Drive\\", "FinnPipeline\\");
        //Files in directories under the parent dir that have "allowed extensions" plit needs a random number generator for reproducibility when splitting the files into train and test
        FileSplit filesInDir = new FileSplit(parentDir, allowedExtensions, randNumGen);

        //You do not have to manually specify labels. This class (instantiated as below) will
        //parse the parent dir and use the name of the subdirectories as label/class names
        labelMaker = new ParentPathLabelGenerator();
        //The balanced path filter gives you fine tune control of the min/max cases to load for each class
        //Below is a bare bones version. Refer to javadocs for details
        BalancedPathFilter pathFilter = new BalancedPathFilter(randNumGen, allowedExtensions, labelMaker);

        //Split the image files into train and test. Specify the train test split as 80%,20%
        InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, 80, 20);
        trainData = filesInDirSplit[0];
        testData = filesInDirSplit[1];

        //Specifying a new record reader with the height and width you want the images to be resized to.
        //Note that the images in this example are all of different size
        //They will all be resized to the height and width specified below
        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);

        //Initialize the record reader with the train data and the transform chain
        recordReader.initialize(trainData);
        //convert the record reader to an iterator for training - Refer to other examples for how to use an iterator
        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, 10, 1, outputNum);
    }

    public DataSetIterator getTrainData() throws IOException {
        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);
        recordReader.initialize(trainData);
        return new RecordReaderDataSetIterator(recordReader, 10, 1, outputNum);
    }

    public DataSetIterator getTestData() throws IOException {
        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);
        recordReader.initialize(testData);
        return new RecordReaderDataSetIterator(recordReader, 10, 1, outputNum);
    }

}
