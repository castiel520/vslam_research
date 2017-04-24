

#ifndef ORB_SLAM2_EXPORT_H
#define ORB_SLAM2_EXPORT_H

#ifdef ORB_SLAM2_EXPORTS
#define ORB_SLAM2_API __declspec(dllexport)
#else
#define ORB_SLAM2_API __declspec(dllimport)
#endif

#endif