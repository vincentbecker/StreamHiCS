import java.io.File;
import java.net.MalformedURLException;
import java.util.Vector;

import javax.media.MediaLocator;

public class Main {

	public static void main(String[] args) {
		String basePath = "C://Users/Vincent/Desktop/Video/";
		
		//String outputFilename = "C://Users/Vincent/Desktop/out.mov";
		String outputFilename = "file:/c:/Users/Vincent/Desktop/out.mov";
		
		int l = new File(basePath).listFiles().length;
		Vector<String> imgLst = new Vector<String>();
		
		for(int i = 1; i <= l; i++){
			imgLst.add(basePath + "image" + i + ".jpeg");
		}

		JPEGImagesToMovie imageToMovie = new JPEGImagesToMovie();
		MediaLocator oml;
		 if ((oml = JPEGImagesToMovie.createMediaLocator(outputFilename)) == null) {
			        System.err.println("Cannot build media locator from: " + outputFilename);
			        System.exit(0);
			    }
			    int interval = 35;
			    try {
					imageToMovie.doIt(684, 661, (1000 / interval), imgLst, oml);
				} catch (MalformedURLException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
	}

}
