#import "OpenCVTestViewController.h"

#define RECT_UPDATE_INTERVAL 0.02

#define REDRAW_DELTA 10
#import <opencv2/imgproc/imgproc_c.h>
#import <opencv2/objdetect/objdetect.hpp>

@implementation OpenCVTestViewController
@synthesize imageView, overlayLayer, recognizedRect;

- (void)dealloc {
	AudioServicesDisposeSystemSoundID(alertSoundID);
	[imageView dealloc];
	self.overlayLayer = nil;
	[super dealloc];
}

#pragma mark -
#pragma mark OpenCV Support Methods

// NOTE you SHOULD cvReleaseImage() for the return value when end of the code.
- (IplImage *)CreateIplImageFromUIImage:(UIImage *)image {
	CGImageRef imageRef = image.CGImage;

	CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
	IplImage *iplimage = cvCreateImage(cvSize(image.size.width, image.size.height), IPL_DEPTH_8U, 4);
	CGContextRef contextRef = CGBitmapContextCreate(iplimage->imageData, iplimage->width, iplimage->height,
													iplimage->depth, iplimage->widthStep,
													colorSpace, kCGImageAlphaPremultipliedLast|kCGBitmapByteOrderDefault);
	CGContextDrawImage(contextRef, CGRectMake(0, 0, image.size.width, image.size.height), imageRef);
	CGContextRelease(contextRef);
	CGColorSpaceRelease(colorSpace);

	IplImage *ret = cvCreateImage(cvGetSize(iplimage), IPL_DEPTH_8U, 3);
	cvCvtColor(iplimage, ret, CV_RGBA2BGR);
	cvReleaseImage(&iplimage);

	return ret;
}

// NOTE You should convert color mode as RGB before passing to this function
- (UIImage *)UIImageFromIplImage:(IplImage *)image {
	NSLog(@"IplImage (%d, %d) %d bits by %d channels, %d bytes/row %s", image->width, image->height, image->depth, image->nChannels, image->widthStep, image->channelSeq);

	CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
	NSData *data = [NSData dataWithBytes:image->imageData length:image->imageSize];
	CGDataProviderRef provider = CGDataProviderCreateWithCFData((CFDataRef)data);
	CGImageRef imageRef = CGImageCreate(image->width, image->height,
										image->depth, image->depth * image->nChannels, image->widthStep,
										colorSpace, kCGImageAlphaNone|kCGBitmapByteOrderDefault,
										provider, NULL, false, kCGRenderingIntentDefault);
	UIImage *ret = [UIImage imageWithCGImage:imageRef];
	CGImageRelease(imageRef);
	CGDataProviderRelease(provider);
	CGColorSpaceRelease(colorSpace);
	return ret;
}

#pragma mark -
#pragma mark Utilities for intarnal use

- (void)showProgressIndicator:(NSString *)text {
	//[UIApplication sharedApplication].networkActivityIndicatorVisible = YES;
	self.view.userInteractionEnabled = FALSE;
	if(!progressHUD) {
		CGFloat w = 160.0f, h = 120.0f;
		progressHUD = [[UIProgressHUD alloc] initWithFrame:CGRectMake((self.view.frame.size.width-w)/2, (self.view.frame.size.height-h)/2, w, h)];
		[progressHUD setText:text];
		[progressHUD showInView:self.view];
	}
}

- (void)hideProgressIndicator {
	//[UIApplication sharedApplication].networkActivityIndicatorVisible = NO;
	self.view.userInteractionEnabled = TRUE;
	if(progressHUD) {
		[progressHUD hide];
		[progressHUD release];
		progressHUD = nil;

		AudioServicesPlaySystemSound(alertSoundID);
	}
}

- (void)opencvEdgeDetect {
	NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];

	if(imageView.image) {
		cvSetErrMode(CV_ErrModeParent);

		// Create grayscale IplImage from UIImage
		IplImage *img_color = [self CreateIplImageFromUIImage:imageView.image];
		IplImage *img = cvCreateImage(cvGetSize(img_color), IPL_DEPTH_8U, 1);
		cvCvtColor(img_color, img, CV_BGR2GRAY);
		cvReleaseImage(&img_color);
		
		// Detect edge
		IplImage *img2 = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
		cvCanny(img, img2, 64, 128, 3);
		cvReleaseImage(&img);
		
		// Convert black and whilte to 24bit image then convert to UIImage to show
		IplImage *image = cvCreateImage(cvGetSize(img2), IPL_DEPTH_8U, 3);
		for(int y=0; y<img2->height; y++) {
			for(int x=0; x<img2->width; x++) {
				char *p = image->imageData + y * image->widthStep + x * 3;
				*p = *(p+1) = *(p+2) = img2->imageData[y * img2->widthStep + x];
			}
		}
		cvReleaseImage(&img2);
		imageView.image = [self UIImageFromIplImage:image];
		cvReleaseImage(&image);

		[self hideProgressIndicator];
	}

	[pool release];
}

- (void) opencvFaceDetect:(UIImage *)overlayImage  {
	NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];

	if(imageView.image) {
		
		int scale = 2;
		IplImage *image = [self CreateIplImageFromUIImage:imageView.image];
		
		CvMemStorage* storage = cvCreateMemStorage(0);
		CvSeq *faces = [self opencvFaceDetectImage:image withOverlay:overlayImage doRotate:NO numChannels:3 withStorage:storage];
	
                /*        
		// Detect faces and draw rectangle on them
		CvSeq* faces = cvHaarDetectObjects(small_image, cascade, storage, 1.2f, 2, CV_HAAR_DO_CANNY_PRUNING, cvSize(0,0), cvSize(20, 20));
		cvReleaseImage(&small_image);
                */
		
		// Create canvas to show the results
		CGImageRef imageRef = imageView.image.CGImage;
		CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
		CGContextRef contextRef = CGBitmapContextCreate(NULL, imageView.image.size.width, imageView.image.size.height,
														8, imageView.image.size.width * 4,
														colorSpace, kCGImageAlphaPremultipliedLast|kCGBitmapByteOrderDefault);
		CGContextDrawImage(contextRef, CGRectMake(0, 0, imageView.image.size.width, imageView.image.size.height), imageRef);
		
		CGContextSetLineWidth(contextRef, 4);
		CGContextSetRGBStrokeColor(contextRef, 0.0, 0.0, 1.0, 0.5);
		
		
		foundFace = (faces->total >=1)?YES:NO;
		// Draw results on the iamge
		for(int i = 0; i < faces->total; i++) {
			NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];
			
			// Calc the rect of faces
			CvRect cvrect = *(CvRect*)cvGetSeqElem(faces, i);
			CGRect face_rect = CGContextConvertRectToDeviceSpace(contextRef, CGRectMake(cvrect.x * scale, cvrect.y * scale, cvrect.width * scale, cvrect.height * scale));
			self.recognizedRect = face_rect;
			if(overlayImage) {
				CGContextDrawImage(contextRef, face_rect, overlayImage.CGImage);
			} else {
				CGContextStrokeRect(contextRef, face_rect);
			}
			
			[pool release];
		}
		
		
		
		imageView.image = [UIImage imageWithCGImage:CGBitmapContextCreateImage(contextRef)];
		CGContextRelease(contextRef);
		CGColorSpaceRelease(colorSpace);
		
		cvReleaseMemStorage(&storage);
		
		
		[self hideProgressIndicator];
	 
	 } 
}

- (CvSeq *)opencvFaceDetectImage:(IplImage *)image withOverlay:(UIImage *)overlayImage doRotate:(BOOL)rotate numChannels:(int)channels withStorage:(CvMemStorage *)storage {

	cvSetErrMode(CV_ErrModeParent);

	
	// Scaling down
	IplImage *small_image = cvCreateImage(cvSize(image->width/2,image->height/2), IPL_DEPTH_8U, channels);
	IplImage *small_image_rotated;
	cvPyrDown(image, small_image, CV_GAUSSIAN_5x5);
	
	if(rotate) {
		small_image_rotated = [self rotateImage:small_image withDegrees:90.0];
	}
		
	// Load XML
	NSString *path = [[NSBundle mainBundle] pathForResource:@"haarcascade_frontalface_default" ofType:@"xml"];
	
	CvHaarClassifierCascade* cascade = (CvHaarClassifierCascade*)cvLoad([path cStringUsingEncoding:NSASCIIStringEncoding], NULL, NULL, NULL);

	
	// Detect faces and draw rectangle on them	
	CvSeq* faces;
	if(rotate) {
		faces = cvHaarDetectObjects(small_image_rotated, cascade, storage, 1.2f, 2, CV_HAAR_DO_CANNY_PRUNING, cvSize(20, 20));
	} else {
		faces = cvHaarDetectObjects(small_image, cascade, storage, 1.2f, 2, CV_HAAR_DO_CANNY_PRUNING, cvSize(20, 20));		
	}
	


	
	cvReleaseImage(&small_image);

	cvReleaseHaarClassifierCascade(&cascade);
	
	if(rotate) {
		cvReleaseImage(&small_image_rotated);
	}
	
	return faces;


}


- (void)updateRecognitionRect {
	// Speed up animations
	//[CATransaction setValue:[NSNumber numberWithFloat:RECT_UPDATE_INTERVAL * 0.4] forKey:kCATransactionAnimationDuration];
	
	CGRect newRect = self.recognizedRect;
	if(foundFace) {
		// only redraw rect for significant changes
		int delta = abs(overlayLayer.frame.origin.x - newRect.origin.x) + abs(overlayLayer.frame.origin.y - newRect.origin.y) +
					abs(overlayLayer.frame.size.width - newRect.size.width) + abs(overlayLayer.frame.size.height - newRect.size.height);
		if(delta > REDRAW_DELTA) {
			NSLog(@"redrawing x : %f y : %f w : %f h : %f", newRect.origin.x, newRect.origin.y, newRect.size.width, newRect.size.height);
			overlayLayer.frame = newRect;
		} 
		overlayLayer.hidden = NO;
	} else {
		overlayLayer.hidden = YES;
	}
	
	[overlayLayer setNeedsDisplay];
}



- (void)drawLayer:(CALayer *)layer inContext:(CGContextRef)context {
	int border = 0;
	CGContextSetLineWidth(context ,2.0);
	CGContextSetRGBStrokeColor(context, 0.0, 0.8, 1.0, 1.0);
	CGContextStrokeRect(context, CGRectMake(layer.bounds.origin.x + border,
											layer.bounds.origin.y + border,
											layer.bounds.size.width - border * 2,
											layer.bounds.size.height - border * 2));
	
}



// Rotate the image clockwise (or counter-clockwise if negative).
// Remember to free the returned image.
- (IplImage *)rotateImage:(IplImage *)src withDegrees:(float )angleDegrees
{
	
	
	IplImage *dest;
	dest = cvCloneImage( src );
	dest->origin = src->origin;
	cvZero( dest );
	
	
	CvPoint2D32f center = cvPoint2D32f(src->width/2, src->height/2);
	double angle = -90.0;
	double scale = 1.0;
	CvMat* rot_mat = cvCreateMat(2,3,CV_32FC1);
	
	cv2DRotationMatrix( center, angle, scale, rot_mat );
	
	cvWarpAffine( src, dest, rot_mat, CV_WARP_FILL_OUTLIERS, cvScalarAll(0) );
	
	return dest;

	[pool release];
}


#pragma mark -
#pragma mark IBAction

- (IBAction)loadImage:(id)sender {
	if(!actionSheetAction) {
		UIActionSheet *actionSheet = [[UIActionSheet alloc] initWithTitle:@""
																 delegate:self cancelButtonTitle:@"Cancel" destructiveButtonTitle:nil
														otherButtonTitles:@"Use Photo from Library", @"Take Photo with Camera", @"Use Default Lena", @"Realtime Camera Mode", nil];
		actionSheet.actionSheetStyle = UIActionSheetStyleDefault;
		actionSheetAction = ActionSheetToSelectTypeOfSource;
		[actionSheet showInView:self.view];
		[actionSheet release];
	}
}

- (IBAction)saveImage:(id)sender {
	if(imageView.image) {
		[self showProgressIndicator:@"Saving"];
		UIImageWriteToSavedPhotosAlbum(imageView.image, self, @selector(finishUIImageWriteToSavedPhotosAlbum:didFinishSavingWithError:contextInfo:), nil);
	}
}

- (void)finishUIImageWriteToSavedPhotosAlbum:(UIImage *)image didFinishSavingWithError:(NSError *)error contextInfo:(void *)contextInfo {
	[self hideProgressIndicator];
}

- (IBAction)edgeDetect:(id)sender {
	[self showProgressIndicator:@"Detecting"];
	[self performSelectorInBackground:@selector(opencvEdgeDetect) withObject:nil];
}

- (IBAction)faceDetect:(id)sender {
	cvSetErrMode(CV_ErrModeParent);
	if(imageView.image && !actionSheetAction) {
		UIActionSheet *actionSheet = [[UIActionSheet alloc] initWithTitle:@""
																 delegate:self cancelButtonTitle:@"Cancel" destructiveButtonTitle:nil
														otherButtonTitles:@"Bounding Box", @"Laughing Man", nil];
		actionSheet.actionSheetStyle = UIActionSheetStyleDefault;
		actionSheetAction = ActionSheetToSelectTypeOfMarks;
		[actionSheet showInView:self.view];
		[actionSheet release];
	}
}

#pragma mark -
#pragma mark UIViewControllerDelegate

- (void)viewDidLoad {
	[super viewDidLoad];
	[[UIApplication sharedApplication] setStatusBarStyle:UIStatusBarStyleBlackOpaque animated:YES];
	[self loadImage:nil];

	NSURL *url = [NSURL fileURLWithPath:[[NSBundle mainBundle] pathForResource:@"Tink" ofType:@"aiff"] isDirectory:NO];
	AudioServicesCreateSystemSoundID((CFURLRef)url, &alertSoundID);
}

- (BOOL)shouldAutorotateToInterfaceOrientation:(UIInterfaceOrientation)interfaceOrientation {
	return NO;
}

#pragma mark -
#pragma mark UIActionSheetDelegate

- (void)actionSheet:(UIActionSheet *)actionSheet clickedButtonAtIndex:(NSInteger)buttonIndex {
	switch(actionSheetAction) {
		case ActionSheetToSelectTypeOfSource: {
			UIImagePickerControllerSourceType sourceType;
			if (buttonIndex == 0) {
				sourceType = UIImagePickerControllerSourceTypePhotoLibrary;
			} else if(buttonIndex == 1) {
				sourceType = UIImagePickerControllerSourceTypeCamera;
			} else if(buttonIndex == 2) {
				NSString *path = [[NSBundle mainBundle] pathForResource:@"lena" ofType:@"jpg"];
				imageView.image = [UIImage imageWithContentsOfFile:path];
				break;
			} else if(buttonIndex==3) {
				[self startCameraCapture];
				break;
			} else {
				// Cancel
				break;
			}
			if([UIImagePickerController isSourceTypeAvailable:sourceType]) {
				UIImagePickerController *picker = [[UIImagePickerController alloc] init];
				picker.sourceType = sourceType;
				picker.delegate = self;
				picker.allowsImageEditing = NO;
				[self presentModalViewController:picker animated:YES];
				[picker release];
			}
			break;
		}
		case ActionSheetToSelectTypeOfMarks: {
			if(buttonIndex != 0 && buttonIndex != 1) {
				break;
			}

			UIImage *image = nil;
			if(buttonIndex == 1) {
				NSString *path = [[NSBundle mainBundle] pathForResource:@"laughing_man" ofType:@"png"];
				image = [UIImage imageWithContentsOfFile:path];
			}

			[self showProgressIndicator:@"Detecting"];
			[self performSelectorInBackground:@selector(opencvFaceDetect:) withObject:image];
			break;
		}
	}
	actionSheetAction = 0;
}

#pragma mark -
#pragma mark UIImagePickerControllerDelegate

- (UIImage *)scaleAndRotateImage:(UIImage *)image {
	static int kMaxResolution = 640;
	
	CGImageRef imgRef = image.CGImage;
	CGFloat width = CGImageGetWidth(imgRef);
	CGFloat height = CGImageGetHeight(imgRef);
	
	CGAffineTransform transform = CGAffineTransformIdentity;
	CGRect bounds = CGRectMake(0, 0, width, height);
	if (width > kMaxResolution || height > kMaxResolution) {
		CGFloat ratio = width/height;
		if (ratio > 1) {
			bounds.size.width = kMaxResolution;
			bounds.size.height = bounds.size.width / ratio;
		} else {
			bounds.size.height = kMaxResolution;
			bounds.size.width = bounds.size.height * ratio;
		}
	}
	
	CGFloat scaleRatio = bounds.size.width / width;
	CGSize imageSize = CGSizeMake(CGImageGetWidth(imgRef), CGImageGetHeight(imgRef));
	CGFloat boundHeight;
	
	UIImageOrientation orient = image.imageOrientation;
	switch(orient) {
		case UIImageOrientationUp:
			transform = CGAffineTransformIdentity;
			break;
		case UIImageOrientationUpMirrored:
			transform = CGAffineTransformMakeTranslation(imageSize.width, 0.0);
			transform = CGAffineTransformScale(transform, -1.0, 1.0);
			break;
		case UIImageOrientationDown:
			transform = CGAffineTransformMakeTranslation(imageSize.width, imageSize.height);
			transform = CGAffineTransformRotate(transform, M_PI);
			break;
		case UIImageOrientationDownMirrored:
			transform = CGAffineTransformMakeTranslation(0.0, imageSize.height);
			transform = CGAffineTransformScale(transform, 1.0, -1.0);
			break;
		case UIImageOrientationLeftMirrored:
			boundHeight = bounds.size.height;
			bounds.size.height = bounds.size.width;
			bounds.size.width = boundHeight;
			transform = CGAffineTransformMakeTranslation(imageSize.height, imageSize.width);
			transform = CGAffineTransformScale(transform, -1.0, 1.0);
			transform = CGAffineTransformRotate(transform, 3.0 * M_PI / 2.0);
			break;
		case UIImageOrientationLeft:
			boundHeight = bounds.size.height;
			bounds.size.height = bounds.size.width;
			bounds.size.width = boundHeight;
			transform = CGAffineTransformMakeTranslation(0.0, imageSize.width);
			transform = CGAffineTransformRotate(transform, 3.0 * M_PI / 2.0);
			break;
		case UIImageOrientationRightMirrored:
			boundHeight = bounds.size.height;
			bounds.size.height = bounds.size.width;
			bounds.size.width = boundHeight;
			transform = CGAffineTransformMakeScale(-1.0, 1.0);
			transform = CGAffineTransformRotate(transform, M_PI / 2.0);
			break;
		case UIImageOrientationRight:
			boundHeight = bounds.size.height;
			bounds.size.height = bounds.size.width;
			bounds.size.width = boundHeight;
			transform = CGAffineTransformMakeTranslation(imageSize.height, 0.0);
			transform = CGAffineTransformRotate(transform, M_PI / 2.0);
			break;
		default:
			[NSException raise:NSInternalInconsistencyException format:@"Invalid image orientation"];
	}
	
	UIGraphicsBeginImageContext(bounds.size);
	CGContextRef context = UIGraphicsGetCurrentContext();
	if (orient == UIImageOrientationRight || orient == UIImageOrientationLeft) {
		CGContextScaleCTM(context, -scaleRatio, scaleRatio);
		CGContextTranslateCTM(context, -height, 0);
	} else {
		CGContextScaleCTM(context, scaleRatio, -scaleRatio);
		CGContextTranslateCTM(context, 0, -height);
	}
	CGContextConcatCTM(context, transform);
	CGContextDrawImage(UIGraphicsGetCurrentContext(), CGRectMake(0, 0, width, height), imgRef);
	UIImage *imageCopy = UIGraphicsGetImageFromCurrentImageContext();
	UIGraphicsEndImageContext();
	
	return imageCopy;
}

- (void)imagePickerController:(UIImagePickerController *)picker
		didFinishPickingImage:(UIImage *)image
				  editingInfo:(NSDictionary *)editingInfo
{
	imageView.image = [self scaleAndRotateImage:image];
	[[picker parentViewController] dismissModalViewControllerAnimated:YES];
}

- (void)imagePickerControllerDidCancel:(UIImagePickerController *)picker {
	[[picker parentViewController] dismissModalViewControllerAnimated:YES];
}



- (void)captureOutput:(AVCaptureOutput *)captureOutput didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer fromConnection:(AVCaptureConnection *)connection {
	int scale = 2;
	IplImage *i = [self createIplImageFromSampleBuffer:sampleBuffer];
	CvMemStorage* storage = cvCreateMemStorage(0);
	CvSeq *faces = [self opencvFaceDetectImage:i withOverlay:NULL doRotate:YES numChannels:4 withStorage:storage];
	
	
	if(faces->total > 0) {
		NSLog(@"found %i iw : %i fw : %i", faces->total, i->width, self.view.frame.size.width);
		// Create canvas to show the results
		CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
		CGContextRef contextRef = CGBitmapContextCreate(NULL, i->width, i->height,
														8, i->width * 4,
														colorSpace, kCGImageAlphaPremultipliedLast|kCGBitmapByteOrderDefault);		
		CvRect cvrect = *(CvRect*)cvGetSeqElem(faces, 0);
		self.recognizedRect = CGRectMake(240-cvrect.x* scale, cvrect.y*scale, cvrect.width*scale, cvrect.height*scale); //CGContextConvertRectToDeviceSpace(contextRef, CGRectMake(cvrect.x * scale, -cvrect.y * scale, cvrect.width * scale, cvrect.height * scale));
		foundFace = YES;
	} else {
		foundFace = NO;
		NSLog(@"did not find faces");
	}

}

- (void)startCameraCapture {
	// Create the AVCapture Session
	NSLog(@"startcameracapture");
//#if HAS_VIDEO_CAPTURE	
	AVCaptureSession *session = [[AVCaptureSession alloc] init];
	
	// Create a preview layer to show the output from the camera
	AVCaptureVideoPreviewLayer *previewLayer = [AVCaptureVideoPreviewLayer layerWithSession:session];
	CGRect f = self.view.frame;
	previewLayer.frame = CGRectMake(f.origin.x, f.origin.y-20, f.size.width, f.size.height);
	previewLayer.videoGravity = AVLayerVideoGravityResizeAspectFill;
	[self.view.layer addSublayer:previewLayer];
	
	
	// Create recognition rectangle overlay
	overlayLayer = [CALayer layer];
	CGRect frame = self.view.layer.bounds;
	frame.origin.x += 10.0f;
	frame.origin.y += 10.0f;
	frame.size.width = 20.0f;
	frame.size.height = 20.0f;
	overlayLayer.frame = frame;
	overlayLayer.backgroundColor = [[UIColor clearColor] CGColor];
	overlayLayer.delegate = self;
	[self.view.layer addSublayer:overlayLayer];
	
	
	AVCaptureDevice *camera = [self frontFacingCamera];
	
	// Create a AVCaptureInput with the camera device
	NSError *error = nil;
	AVCaptureInput *cameraInput = [[AVCaptureDeviceInput alloc] initWithDevice:camera error:&error];
	if(cameraInput == nil) {
		NSLog(@"Error to create camera capture:%@", error);
	}
	
	// Set the outpua
	AVCaptureVideoDataOutput *videoOutput = [[AVCaptureVideoDataOutput alloc] init];
	videoOutput.alwaysDiscardsLateVideoFrames = YES;
	
	// create a queue besides the main thread queue to run the capture on
	dispatch_queue_t captureQueue = dispatch_queue_create("captureQueue", NULL);
	
	// setup our delegate
	[videoOutput setSampleBufferDelegate:self queue:captureQueue];
	
	// release the queue.  I still don't entirely understand why we're release it here,
	// but the code examples I've found indicate this is the right thing. 
	dispatch_release(captureQueue);
	
	// configure the pixel format
	
	[videoOutput setVideoSettings:[NSDictionary dictionaryWithObject:[NSNumber numberWithInt:kCVPixelFormatType_32BGRA] forKey:(id)kCVPixelBufferPixelFormatTypeKey]]; // BGRA is necessary for manual preview
	
	// and the size of the frames we want
	// try AVCaptureSessionPresetLow if this is too slow...
	[session setSessionPreset:AVCaptureSessionPresetMedium];
	
	// If you wish to cap the frame rate to a known value, such as 10 fps, set 
	// minFrameDuration.
	videoOutput.minFrameDuration = CMTimeMake(1, 10);
	
	// Add the input and output
	[session addInput:cameraInput];
	[session addOutput:videoOutput];
	
	// Start the session
	[session startRunning];  
	
	
	[NSTimer scheduledTimerWithTimeInterval:RECT_UPDATE_INTERVAL target:self selector:@selector(updateRecognitionRect) userInfo:nil repeats:YES];
	
//#endif		
	
}



- (IplImage *)createIplImageFromSampleBuffer:(CMSampleBufferRef)sampleBuffer {
    IplImage *iplimage = 0;
    if (sampleBuffer) {
        CVImageBufferRef imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
        CVPixelBufferLockBaseAddress(imageBuffer, 0);
		
        // get information of the image in the buffer
        uint8_t *bufferBaseAddress = (uint8_t *)CVPixelBufferGetBaseAddressOfPlane(imageBuffer, 0);
        size_t bufferWidth = CVPixelBufferGetWidth(imageBuffer);
        size_t bufferHeight = CVPixelBufferGetHeight(imageBuffer);
		
        // create IplImage
        if (bufferBaseAddress) {
            iplimage = cvCreateImage(cvSize(bufferWidth, bufferHeight), IPL_DEPTH_8U, 4);
            iplimage->imageData = (char*)bufferBaseAddress;
        }
		
        // release memory
        CVPixelBufferUnlockBaseAddress(imageBuffer, 0);
    }
    else
        NSLog(@"No sampleBuffer!!");
	
    return iplimage;
}

- (AVCaptureDevice *)frontFacingCamera {
	NSArray *cameras = [AVCaptureDevice devicesWithMediaType:AVMediaTypeVideo];
	for (AVCaptureDevice *device in cameras) {
		if(device.position == AVCaptureDevicePositionFront) {
			return device;
		}
	}
	return [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeVideo];
}


@end
