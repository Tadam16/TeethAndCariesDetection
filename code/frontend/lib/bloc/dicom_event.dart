part of 'dicom_bloc.dart';

/// Events

@immutable
abstract class DicomEvent {
  const DicomEvent();
}

class LoadDicomEvent extends DicomEvent {
  static final LoadDicomEvent _instance = LoadDicomEvent._();

  factory LoadDicomEvent() => _instance;

  LoadDicomEvent._();
}
