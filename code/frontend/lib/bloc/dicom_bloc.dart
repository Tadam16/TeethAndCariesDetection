import 'package:bloc/bloc.dart';
import 'package:flutter/cupertino.dart';
import 'package:equatable/equatable.dart';

part 'dicom_event.dart';
part 'dicom_state.dart';

/// Questionnaire BLoC is an element of BLoC pattern in MVVM design pattern
/// Manages the QuestionnairePage events and states async

class DicomBloc extends Bloc<DicomEvent, DicomState> {
  final _interactor;
  DicomBloc(this._interactor) : super(Loading());

  /// This method emits states from incoming events
  @override
  Stream<DicomState> mapEventToState(DicomEvent event) async* {
    if (event is LoadDicomEvent) {
      yield* _mapLoadDicomToState();
    }
  }

  Stream<DicomState> _mapLoadDicomToState() async* {
    try {
      print("getting Photos");
      final photos = await _interactor.getPhotoList();
      print("Successful Http request!");
      yield ContentReady(photos: photos);
    } on Exception catch (e) {
      print("http fail: ${e.toString()}");
      yield Error(photos: const []);
    }
  }
}
