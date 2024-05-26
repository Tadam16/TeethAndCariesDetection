import 'dart:async';
import 'package:dio/dio.dart';
import 'package:pretty_dio_logger/pretty_dio_logger.dart';
import 'package:retrofit/retrofit.dart';

import 'dicom_api.dart';


part 'dicom_api_service.g.dart';

@RestApi(baseUrl: "https://localhost:3456")
abstract class DicomApiService implements DicomApi {
  factory DicomApiService() {
    final _dio = Dio();
    _dio.interceptors.add(LogInterceptor());
    //_dio.interceptors.add(RequestInterceptor());
    _dio.interceptors.add(PrettyDioLogger(
      requestHeader: true,
      requestBody: true,
      responseBody: false,
      responseHeader: true,
      compact: false,
    ));

    return _DicomApiService(_dio);
  }


  @override
  @POST("/")
  @MultiPart()
  Future<HttpResponse<dynamic>> upload(
      @Part(contentType: "image/jpeg") Map<String, MultipartFile> photos,
      );
}

