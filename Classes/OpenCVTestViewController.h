#import <UIKit/UIKit.h>
#import <AudioToolbox/AudioToolbox.h>

#import <CoreMedia/CoreMedia.h>
#import <QuartzCore/QuartzCore.h>
#import <AVFoundation/AVFoundation.h>
#import <CoreVideo/CoreVideo.h>
#import <opencv/cv.h>

#define HAS_VIDEO_CAPTURE (__IPHONE_OS_VERSION_MIN_REQUIRED >= 40000 && TARGET_OS_EMBEDDED)

@interface UIProgressIndicator : UIActivityIndicatorView  {
}

+ (struct CGSize)size;
- (int)progressIndicatorStyle;
- (void)setProgressIndicatorStyle:(int)fp8;
- (void)setStyle:(int)fp8;
- (void)setAnimating:(BOOL)fp8;
- (void)startAnimation;
- (void)stopAnimation;
@end

@interface UIProgressHUD : UIView {
    UIProgressIndicator *_progressIndicator;
    UILabel *_progressMessage;
    UIImageView *_doneView;
    UIWindow *_parentWindow;
    struct {
        unsigned int isShowing:1;
        unsigned int isShowingText:1;
        unsigned int fixedFrame:1;
        unsigned int reserved:30;
    } _progressHUDFlags;
}

- (id)_progressIndicator;
- (id)initWithFrame:(struct CGRect)fp8;
- (void)setText:(id)fp8;
- (void)setShowsText:(BOOL)fp8;
- (void)setFontSize:(int)fp8;
- (void)drawRect:(struct CGRect)fp8;
- (void)layoutSubviews;
- (void)showInView:(id)fp8;
- (void)hide;
- (void)done;
- (void)dealloc;


@end

typedef enum {
    ActionSheetToSelectTypeOfSource = 1,
    ActionSheetToSelectTypeOfMarks
} OpenCVTestViewControllerActionSheetAction;

@interface OpenCVTestViewController : UIViewController <UIActionSheetDelegate, UIImagePickerControllerDelegate, UINavigationControllerDelegate, AVCaptureVideoDataOutputSampleBufferDelegate> {
	IBOutlet UIImageView *imageView;
	OpenCVTestViewControllerActionSheetAction actionSheetAction;
	UIProgressHUD *progressHUD;
	SystemSoundID alertSoundID;
	CALayer *overlayLayer;
	CGRect recognizedRect;
	BOOL foundFace;
}

- (IBAction)loadImage:(id)sender;
- (IBAction)saveImage:(id)sender;
- (IBAction)edgeDetect:(id)sender;
- (IBAction)faceDetect:(id)sender;



@property (nonatomic, retain) UIImageView *imageView;
@property (nonatomic, retain) CALayer *overlayLayer;
@property (readwrite) CGRect recognizedRect;


- (void)startCameraCapture;
- (AVCaptureDevice *)frontFacingCamera;
- (IplImage *)createIplImageFromSampleBuffer:(CMSampleBufferRef)sampleBuffer;
- (CvSeq *)opencvFaceDetectImage:(IplImage *)image withOverlay:(UIImage *)overlayImage doRotate:(BOOL)rotate numChannels:(int)channels withStorage:(CvMemStorage *)storage;

- (IplImage *)rotateImage:(IplImage *)src withDegrees:(float )angleDegrees;

@end
