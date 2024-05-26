part of 'dicom_bloc.dart';


/// States

@immutable
abstract class DicomState {
  const DicomState();
}

class Loading extends DicomState {
  static final Loading _instance = Loading._();

  factory Loading() => _instance;

  Loading._();
}

class Content extends DicomState {
  final List<Object> photos;

  Content({required this.photos});
}

class ContentReady extends Content with EquatableMixin {
  ContentReady({required List<Object> photos}) : super(photos: photos);

  @override
  List<Object> get props => photos;
}

class Error extends Content with EquatableMixin {
  Error({required List<Object> photos}) : super(photos: photos);

  @override
  List<Object?> get props => photos;
}
